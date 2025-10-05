"""
Hybrid YAML + SLM Refinement Extractor

Two-stage pipeline:
1. Fast YAML extraction (50ms)
2. SLM error correction using a small model (150ms budget)

Provider options:
- MLX (local inference via mlx_lm)
- OpenAI-compatible HTTP (e.g., LM Studio)

Total latency target: <200ms
Expected F1: 0.70-0.80
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .yaml_extractor import YAMLExtractor


class YAMLWithSLMRefinement:
    """
    Hybrid extractor combining YAML patterns with SLM refinement.

    Strategy:
    1. Use YAML for fast initial extraction
    2. Use small language model to fix common errors
    3. Maintain <200ms total latency
    """

    def __init__(
        self,
        yaml_path: Optional[str] = None,
        slm_model: Optional[str] = None,
        max_refinement_ms: int = 150
    ):
        """
        Initialize hybrid extractor.

        Args:
            yaml_path: Path to YAML index file
            slm_model: MLX model path (default: Qwen2.5-0.5B)
            max_refinement_ms: Maximum time for SLM refinement
        """
        # YAML base extractor
        self.yaml_path = yaml_path or os.getenv(
            "YAML_INDEX_PATH",
            "server/archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml"
        )
        self.yaml_extractor = YAMLExtractor(self.yaml_path)

        # SLM configuration
        self.slm_enabled = os.getenv("SLM_REFINEMENT_ENABLED", "true").lower() == "true"
        self.force_refinement = os.getenv("SLM_FORCE", "false").lower() in ("1", "true", "yes")
        if self.force_refinement:
            self.slm_enabled = True

        # Provider: "mlx" or "openai" (LM Studio)
        self.slm_provider = os.getenv("SLM_PROVIDER", "mlx").strip().lower()

        # Models: primary and optional secondary for fallback/dual mode
        if self.slm_provider == "openai":
            # OpenAI-compatible (LM Studio)
            self.slm_primary_model = (
                slm_model
                or os.getenv("SLM_PRIMARY_MODEL")
                or "lfm2-350m-extract"
            )
            self.slm_secondary_model = os.getenv("SLM_SECONDARY_MODEL", "qwen2.5-coder-0.5b-instruct")
            self.slm_mode = os.getenv("SLM_MODE", "single").strip().lower()  # single | dual | fallback
            self.slm_base_url = os.getenv("SLM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
            self.slm_api_key = os.getenv("SLM_API_KEY", os.getenv("OPENAI_API_KEY", "not-needed"))
            self.slm_temperature = float(os.getenv("SLM_TEMP", "0.1"))
            self.slm_max_tokens = int(os.getenv("SLM_MAX_TOKENS", "120"))
            self.prewarm_on_init = os.getenv("SLM_PREWARM_ON_INIT", "true").lower() in ("1", "true", "yes")
        else:
            # MLX (local)
            self.slm_model = slm_model or os.getenv(
                "SLM_MODEL_PATH",
                "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
            )
            self.prewarm_on_init = False

        self.max_refinement_ms = max_refinement_ms

        # Output and validation constraints for SLM
        self.strict_json = os.getenv("SLM_STRICT_JSON", "true").lower() in ("1", "true", "yes")
        allowed_rel = os.getenv(
            "SLM_ALLOWED_RELATIONS",
            "has,works_at,lives_in,founded,discovered,teaches_at,studied_at,"
            "married_to,friend_of,part_of,located_in,leads,manages,owns,is,"
            "has_quality,has_quantity,works_on,focus_on,agree_on,result_in"
        )
        self.allowed_relations = {r.strip() for r in allowed_rel.split(',') if r.strip()}
        self.max_total_triples = int(os.getenv("SLM_MAX_TOTAL_TRIPLES", "24"))
        self.max_added_triples = int(os.getenv("SLM_MAX_ADDED_TRIPLES", "6"))

        # MLX model (lazy loading)
        self._mlx_model = None
        self._mlx_tokenizer = None

        logger.info(
            f"YAMLWithSLMRefinement initialized: "
            f"SLM={self.slm_enabled}, provider={self.slm_provider}"
        )

        # Optional pre-warm (LM Studio)
        if self.slm_enabled and self.slm_provider == "openai" and self.prewarm_on_init:
            try:
                self.prewarm()
            except Exception as e:
                logger.warning(f"SLM pre-warm failed: {e}")

        # Metrics
        self._metrics: Dict[str, Any] = {
            "request_count": 0,
            "json_valid_count": 0,
            "accepted_edit_count": 0,
            "fallback_used_count": 0,
            "unknown_relation_reject_count": 0,
        }

    def _load_mlx_model(self):
        """Lazy load MLX model for SLM refinement."""
        if self._mlx_model is None and self.slm_enabled and self.slm_provider == "mlx":
            try:
                import mlx_lm

                logger.info(f"Loading MLX model: {self.slm_model}")
                self._mlx_model, self._mlx_tokenizer = mlx_lm.load(self.slm_model)
                logger.info("MLX model loaded successfully")
            except ImportError:
                logger.warning("MLX not available, disabling SLM refinement")
                self.slm_enabled = False
            except Exception as e:
                logger.error(f"Failed to load MLX model: {e}")
                self.slm_enabled = False

    def extract(
        self,
        text: str,
        lang: str = "en"
    ) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:
        """
        Extract with YAML and optional SLM refinement.

        Returns same format as YAMLExtractor for compatibility:
            (entities, triples, neg_count, doc)
        """
        # Stage 1: YAML extraction
        t0 = time.perf_counter()
        entities, triples, neg_count, doc = self.yaml_extractor.extract(text, lang)
        triples = self.yaml_extractor.refine(text, triples, doc)
        yaml_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            f"YAML extracted {len(triples)} triples in {yaml_ms:.1f}ms"
        )

        # Stage 2: SLM refinement (time budget enforced inside)
        if self.slm_enabled:
            refined_triples = self._refine_with_slm(text, triples, lang)
            if refined_triples:
                triples = refined_triples

        return entities, triples, neg_count, doc

    def _refine_with_slm(
        self,
        text: str,
        triples: List[Tuple[str, str, str]],
        lang: str
    ) -> Optional[List[Tuple[str, str, str]]]:
        """
        Refine extractions using small language model.

        Focus on common YAML errors:
        - Missing pronoun resolution
        - Incomplete relations
        - Wrong preposition mapping
        """
        # Select provider
        if self.slm_provider == "openai":
            return self._refine_with_openai(text, triples)
        else:
            # MLX path
            self._load_mlx_model()
            if not self._mlx_model:
                return None

    # Public helper for staged policy
    def refine_triples(self, triples: List[Tuple[str, str, str]], text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        """Refine an existing set of triples using the configured SLM provider.
        Falls back to the original triples if refinement fails or is disabled.
        """
        if not self.slm_enabled and not self.force_refinement:
            return triples
        try:
            refined = self._refine_with_slm(text, triples, lang)
            return refined or triples
        except Exception as e:
            logger.debug(f"refine_triples failed: {e}")
            return triples

            try:
                t0 = time.perf_counter()
                prompt = self._build_refinement_prompt(text, triples)
                import mlx_lm
                response = mlx_lm.generate(
                    self._mlx_model,
                    self._mlx_tokenizer,
                    prompt=prompt,
                    max_tokens=min(self.slm_max_tokens if hasattr(self, 'slm_max_tokens') else 100, 256),
                    temp=getattr(self, 'slm_temperature', 0.1),
                    verbose=False
                )
                refined = self._parse_slm_response(response)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.debug(
                    f"SLM refined {len(triples)} → {len(refined)} triples in {elapsed_ms:.1f}ms (mlx)"
                )
                if elapsed_ms < self.max_refinement_ms:
                    return refined
                else:
                    logger.warning(f"SLM refinement too slow: {elapsed_ms:.1f}ms")
                    return None
            except Exception as e:
                logger.warning(f"SLM refinement failed: {e}")
                return None

    def _refine_with_openai(
        self,
        text: str,
        triples: List[Tuple[str, str, str]]
    ) -> Optional[List[Tuple[str, str, str]]]:
        """Refine with OpenAI-compatible API (LM Studio)."""
        import requests

        # Derive dynamic allowed relations from YAML output
        dyn_allowed = set(self.allowed_relations) | {r for _, r, _ in triples}

        def call_model(model: str) -> Optional[List[Tuple[str, str, str]]]:
            t0 = time.perf_counter()
            prompt = self._build_refinement_prompt(text, triples, allowed_relations=dyn_allowed)
            url = f"{self.slm_base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.slm_api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a constrained IE corrector. Follow the schema and constraints exactly. "
                            "Output must be valid JSON with no trailing commentary."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.slm_temperature,
                "max_tokens": self.slm_max_tokens,
            }
            if self.strict_json:
                # Prefer JSON Schema; use flexible schema that accepts arrays or objects per item
                prefer_schema = os.getenv("SLM_RESPONSE_JSON_SCHEMA", "false").lower() in ("1", "true", "yes")
                if prefer_schema:
                    flexible_schema = {
                        "name": "triple_refinement",
                        "strict": True,
                        "schema": {
                            "$schema": "https://json-schema.org/draft/2020-12/schema",
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["triples"],
                            "properties": {
                                "triples": {
                                    "type": "array",
                                    "items": {
                                        "anyOf": [
                                            {
                                                "type": "array",
                                                "minItems": 3,
                                                "maxItems": 3,
                                                "items": [
                                                    {"type": "string", "minLength": 1, "maxLength": 120},
                                                    {"type": "string", "minLength": 1, "maxLength": 64},
                                                    {"type": "string", "minLength": 0, "maxLength": 200}
                                                ]
                                            },
                                            {
                                                "type": "object",
                                                "additionalProperties": False,
                                                "required": ["s", "r", "o"],
                                                "properties": {
                                                    "s": {"type": "string", "minLength": 1, "maxLength": 120},
                                                    "r": {"type": "string", "minLength": 1, "maxLength": 64},
                                                    "o": {"type": "string", "minLength": 0, "maxLength": 200}
                                                }
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    }
                    body["response_format"] = {"type": "json_schema", "json_schema": flexible_schema}
                else:
                    body["response_format"] = {"type": "json_object"}
            try:
                logger.debug(f"SLM HTTP → {url} model={model} strict={self.strict_json}")
                self._metrics["request_count"] += 1
                # Use separate connect/read timeouts; read timeout slightly above budget to avoid disconnects
                connect_to = float(os.getenv("SLM_HTTP_CONNECT_TIMEOUT", "0.5"))
                budget_s = max(self.max_refinement_ms / 1000.0, 0.2)
                read_to = float(os.getenv("SLM_HTTP_READ_TIMEOUT", "0")) or (budget_s + 0.3)
                resp = requests.post(url, headers=headers, json=body, timeout=(connect_to, read_to))
                if resp.status_code != 200:
                    logger.debug(f"SLM HTTP {resp.status_code}: {resp.text[:120]}")
                    return None
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                refined, json_ok = self._parse_slm_response(content, return_valid=True)
                if json_ok:
                    self._metrics["json_valid_count"] += 1
                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.debug(
                    f"SLM refined {len(triples)} → {len(refined)} triples in {elapsed_ms:.1f}ms (openai:{model})"
                )
                if elapsed_ms < self.max_refinement_ms:
                    merged, acc = self._validate_and_merge(triples, refined, allowed_relations=dyn_allowed)
                    if acc.get("unknown_rejects", 0) > 0:
                        self._metrics["unknown_relation_reject_count"] += acc["unknown_rejects"]
                    if acc.get("accepted", False):
                        self._metrics["accepted_edit_count"] += 1
                    else:
                        logger.debug("SLM edits rejected or unchanged; attempting repair if enabled")
                    if not merged or merged == triples:
                        # Attempt one repair pass if nothing changed
                        repair = {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": "Repair: Output only a valid JSON object per schema with key 'triples'."},
                                {"role": "user", "content": self._build_refinement_prompt(text, triples)}
                            ],
                            "temperature": self.slm_temperature,
                            "max_tokens": self.slm_max_tokens,
                        }
                        if self.strict_json:
                            repair["response_format"] = body.get("response_format")
                        try:
                            self._metrics["request_count"] += 1
                            r2 = requests.post(url, headers=headers, json=repair, timeout=max(self.max_refinement_ms / 1000.0, 0.2))
                            if r2.status_code == 200:
                                data2 = r2.json()
                                content2 = data2.get("choices", [{}])[0].get("message", {}).get("content", "")
                                refined2, json_ok2 = self._parse_slm_response(content2, return_valid=True)
                                if json_ok2:
                                    self._metrics["json_valid_count"] += 1
                                merged2, acc2 = self._validate_and_merge(triples, refined2, allowed_relations=dyn_allowed)
                                if acc2.get("unknown_rejects", 0) > 0:
                                    self._metrics["unknown_relation_reject_count"] += acc2["unknown_rejects"]
                                if acc2.get("accepted", False):
                                    self._metrics["accepted_edit_count"] += 1
                                if merged2 and merged2 != triples:
                                    return merged2
                        except Exception:
                            pass
                    return merged
                return None
            except Exception as e:
                logger.debug(f"SLM request failed ({model}): {e}")
                return None

        # Mode handling: single, fallback, dual (prefer first valid)
        if self.slm_mode == "dual":
            out = call_model(self.slm_primary_model)
            if out and out != triples:
                return out
            self._metrics["fallback_used_count"] += 1
            return call_model(self.slm_secondary_model)
        elif self.slm_mode == "fallback":
            out = call_model(self.slm_primary_model)
            if out and out != triples:
                return out
            self._metrics["fallback_used_count"] += 1
            return call_model(self.slm_secondary_model)
        else:
            return call_model(self.slm_primary_model)

    def _build_refinement_prompt(
        self,
        text: str,
        triples: List[Tuple[str, str, str]],
        allowed_relations: Optional[set] = None,
    ) -> str:
        """Build prompt for SLM refinement."""

        # Format current extractions
        current = json.dumps([list(t) for t in triples])

        # Strict prompt with schema, constraints, and few-shot guidance
        schema = {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "array", "items": [{"type": "string"}, {"type": "string"}, {"type": "string"}], "minItems": 3, "maxItems": 3},
                            {"type": "object", "properties": {"s": {"type": "string"}, "r": {"type": "string"}, "o": {"type": "string"}}, "required": ["s", "r", "o"]}
                        ]
                    }
                }
            },
            "required": ["triples"],
            "additionalProperties": False
        }

        use_schema = os.getenv("SLM_RESPONSE_JSON_SCHEMA", "false").lower() in ("1", "true", "yes")
        few_shot = (
            "Example\n"
            "Text: 'I work at OpenAI and live in SF.'\n"
            "Current: [[\"i\",\"work\",\"openai\"]]\n"
            "Corrected: {\"triples\":[[\"you\",\"works_at\",\"openai\"],[\"you\",\"lives_in\",\"sf\"]]}\n"
        )

        allowed = sorted(list((allowed_relations or self.allowed_relations) or []))

        prompt = f"""Fix extraction errors. Output strict JSON only.

Text: "{text}"
Current: {current}

Constraints:
- Output EXACTLY a JSON object with key "triples".
- Each triple is either [subject, relation, object] or an object {{"s":"","r":"","o":""}}.
- Allowed relations only: {allowed}.
- Normalize first person (I, me, my) → "you". Lowercase all text. Keep objects concise.
- Split coordinated facts ("X and Y founded Z" → two triples). Resolve simple pronouns.
- No explanations. No markdown. No trailing text. JSON must parse with no fixes.
- Do not add more than {self.max_added_triples} new triples beyond Current.
- Do not exceed {self.max_total_triples} triples total.

{few_shot if not use_schema else ''}

Return only the JSON object.
"""

        return prompt

    def _parse_slm_response(self, response: str, return_valid: bool = False):
        """Parse SLM response into triples. Accepts object with 'triples' or bare array.
        When return_valid=True, returns (triples, is_json_valid)."""
        text = (response or "").strip()
        # Try direct JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "triples" in data:
                data = data.get("triples", [])
            triples: List[Tuple[str, str, str]] = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, list) and len(item) >= 3:
                        triples.append((str(item[0]), str(item[1]), str(item[2])))
                    elif isinstance(item, dict):
                        s = item.get("s")
                        r = item.get("r")
                        o = item.get("o")
                        if s is not None and r is not None and o is not None:
                            triples.append((str(s), str(r), str(o)))
            if triples:
                out = [self._normalize_triple(t) for t in triples]
                return (out, True) if return_valid else out
        except Exception:
            pass
        # Fallback: scan first JSON array region
        try:
            start = text.find('[')
            end = text.rfind(']')
            if start >= 0 and end > start:
                arr = json.loads(text[start:end+1])
                triples = []
                for item in arr:
                    if isinstance(item, list) and len(item) >= 3:
                        triples.append((str(item[0]), str(item[1]), str(item[2])))
                out = [self._normalize_triple(t) for t in triples]
                return (out, True) if return_valid else out
        except Exception as e:
            logger.debug(f"Failed to parse SLM response: {e}")
        return ([], False) if return_valid else []

    def _normalize_triple(self, t: Tuple[str, str, str]) -> Tuple[str, str, str]:
        s, r, o = (str(t[0]).strip().lower(), str(t[1]).strip().lower(), str(t[2]).strip().lower())
        if s in {"i", "me", "my", "mine", "myself"}:
            s = "you"
        rel_map = {
            "work_at": "works_at", "work_in": "works_in", "work_on": "works_on",
            "live_in": "lives_in", "live_at": "lives_in",
            "focus_on": "focus_on", "agree_on": "agree_on", "result_in": "result_in",
        }
        r = rel_map.get(r, r)
        return (s, r, o)

    def _validate_and_merge(
        self,
        current: List[Tuple[str, str, str]],
        proposed: List[Tuple[str, str, str]],
        allowed_relations: Optional[set] = None,
    ) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any]]:
        if not proposed:
            return current, {"accepted": False, "unknown_rejects": 0}
        valid: List[Tuple[str, str, str]] = []
        use_allowed = set(self.allowed_relations)
        if allowed_relations:
            use_allowed |= set(allowed_relations)
        unknown_rejects = 0
        for s, r, o in proposed:
            if not s or not r or o is None:
                continue
            if use_allowed and r not in use_allowed and r not in {"is"}:
                unknown_rejects += 1
                continue
            if len(s) > 120 or len(o) > 200:
                continue
            valid.append((s, r, o))
        if not valid:
            return current, {"accepted": False, "unknown_rejects": unknown_rejects}
        base_set = set(current)
        merged = list({*base_set, *valid})
        added = [t for t in merged if t not in base_set]
        accepted = False
        if len(added) > self.max_added_triples:
            # Prefer edits that touch existing entities
            base_entities = set()
            for s, _, o in current:
                base_entities.add(s)
                base_entities.add(o)
            def score(t):
                s, _, o = t
                return int(s in base_entities) + int(o in base_entities)
            added_sorted = sorted(added, key=score, reverse=True)[: self.max_added_triples]
            merged = list(base_set) + added_sorted
            accepted = bool(added_sorted)
        else:
            accepted = bool(added)
        if len(merged) > self.max_total_triples:
            merged = merged[: self.max_total_triples]
        return merged, {"accepted": accepted, "unknown_rejects": unknown_rejects}

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        base = {
            "slm_enabled": self.slm_enabled,
            "provider": self.slm_provider,
            "max_refinement_ms": self.max_refinement_ms,
        }
        if self.slm_provider == "openai":
            base.update({
                "primary_model": getattr(self, "slm_primary_model", None),
                "secondary_model": getattr(self, "slm_secondary_model", None),
                "mode": getattr(self, "slm_mode", None),
            })
        else:
            base.update({
                "slm_model": getattr(self, "slm_model", None),
            })
        base.update({"metrics": dict(self._metrics)})
        return base

    def prewarm(self) -> None:
        """Pre-warm SLM provider/models to reduce first-call latency."""
        if not self.slm_enabled:
            return
        if self.slm_provider == "openai":
            import requests
            url = f"{self.slm_base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.slm_api_key}",
                "Content-Type": "application/json",
            }
            models = [self.slm_primary_model]
            if self.slm_mode in ("dual", "fallback") and self.slm_secondary_model:
                models.append(self.slm_secondary_model)
            for m in models:
                body = {
                    "model": m,
                    "messages": [
                        {"role": "system", "content": "Warm-up call. Return [] only."},
                        {"role": "user", "content": "[]"},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1,
                }
                try:
                    requests.post(url, headers=headers, json=body, timeout=1.5)
                    logger.info(f"SLM pre-warmed model: {m}")
                except Exception as e:
                    logger.debug(f"Pre-warm failed for {m}: {e}")
        else:
            # MLX: trigger lazy load
            self._load_mlx_model()


class AsyncYAMLWithSLMRefinement(YAMLWithSLMRefinement):
    """Async version for better integration with voice pipeline."""

    async def extract_async(
        self,
        text: str,
        lang: str = "en"
    ) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:
        """Async extraction with parallel SLM refinement."""

        # Run YAML extraction
        loop = asyncio.get_event_loop()
        entities, triples, neg_count, doc = await loop.run_in_executor(
            None,
            self.yaml_extractor.extract,
            text,
            lang
        )

        # Refine in executor
        triples = await loop.run_in_executor(
            None,
            self.yaml_extractor.refine,
            text,
            triples,
            doc
        )

        # SLM refinement in background
        if self.slm_enabled:
            refined = await loop.run_in_executor(
                None,
                self._refine_with_slm,
                text,
                triples,
                lang
            )
            if refined:
                triples = refined

        return entities, triples, neg_count, doc


def create_hybrid_extractor(
    yaml_path: Optional[str] = None,
    **kwargs
) -> YAMLWithSLMRefinement:
    """Factory function for creating hybrid extractor."""

    # Check if async is needed
    if kwargs.get("async_mode", False):
        return AsyncYAMLWithSLMRefinement(yaml_path, **kwargs)

    return YAMLWithSLMRefinement(yaml_path, **kwargs)
