import os
import torch
import torch.nn as nn
import timm


class SpectreTrack(nn.Module):
    def __init__(
        self,
        model_name="vit_base_patch16_224.mae",
        checkpoint_path="/root/victor/vit_base_patch16_224.mae.pth",
        pretrained=True,
        num_heads=12,
        dropout=0.1,
    ):
        super().__init__()

        try:
            print(f"[INFO] Loading local checkpoint: {checkpoint_path}")

            # Create backbone WITHOUT pretrained weights
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                global_pool="",
                dynamic_img_size=True,
            )

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    checkpoint = checkpoint["model"]

            self.backbone.load_state_dict(checkpoint)

            print("[INFO] Local checkpoint loaded successfully.")

        except Exception as e:
            print(f"[WARNING] Local load failed: {e}")
            print("[INFO] Falling back to timm pretrained weights...")

            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
                global_pool="",
                dynamic_img_size=True,
            )

        self.embed_dim = self.backbone.embed_dim

        self.template_to_search_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.history_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_search = nn.LayerNorm(self.embed_dim)
        self.norm_history = nn.LayerNorm(self.embed_dim)
        self.norm_mlp = nn.LayerNorm(self.embed_dim)

        self.token_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(dropout),
        )

        self.bbox_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4),
        )

        self.confidence_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def _extract_tokens(self, image):
        tokens = self.backbone.forward_features(image)

        # ViT/timm usually returns [B, 1 + N, C], where token 0 is CLS.
        if tokens.ndim == 3 and tokens.shape[1] > 1:
            tokens = tokens[:, 1:, :]

        return tokens

    def extract_clean_features(self, template, search_clean):
        template_tokens = self._extract_tokens(template)
        search_tokens = self._extract_tokens(search_clean)

        attended, _ = self.template_to_search_attn(
            query=search_tokens,
            key=template_tokens,
            value=template_tokens,
            need_weights=False,
        )

        search_tokens = self.norm_search(search_tokens + attended)
        search_tokens = self.norm_mlp(search_tokens + self.token_mlp(search_tokens))

        return search_tokens

    def forward(self, template, search_input, history_buffer=None):
        template_tokens = self._extract_tokens(template)
        search_tokens = self._extract_tokens(search_input)

        template_attended, _ = self.template_to_search_attn(
            query=search_tokens,
            key=template_tokens,
            value=template_tokens,
            need_weights=False,
        )

        search_tokens = self.norm_search(search_tokens + template_attended)

        if history_buffer is not None:
            history_attended, _ = self.history_attn(
                query=search_tokens,
                key=history_buffer,
                value=history_buffer,
                need_weights=False,
            )
            search_tokens = self.norm_history(search_tokens + history_attended)

        search_tokens = self.norm_mlp(search_tokens + self.token_mlp(search_tokens))

        pooled = search_tokens.mean(dim=1)

        bbox = torch.sigmoid(self.bbox_head(pooled))
        confidence = torch.sigmoid(self.confidence_head(pooled)).squeeze(-1)

        return bbox, search_tokens, confidence