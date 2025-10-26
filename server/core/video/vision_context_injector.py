"""Vision context injector - adds video frames to LLM context for streaming vision."""
import asyncio
import os
from loguru import logger
from PIL import Image
from pipecat.frames.frames import Frame, InputImageRawFrame, TextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.pipeline.pipeline import FrameDirection
from typing import Optional


class VisionContextInjector(FrameProcessor):
    """
    Injects video frames into the LLM context for streaming vision.

    For continuous camera input, this processor intercepts InputImageRawFrame
    and directly adds it to the OpenAILLMContext using add_image_frame_message().

    This is different from UserImageRawFrame which is designed for function-call-based vision.
    """

    def __init__(self, context, target_fps: float = 2.0, inject_on_text: bool = True,
                 keyword_filter: bool = False, keywords: Optional[list] = None,
                 on_camera_state_change: Optional[callable] = None):
        """
        Args:
            context: OpenAILLMContext instance to inject images into
            target_fps: Frame rate for throttling (default 2fps)
            inject_on_text: Only inject images when user sends text (default True)
            keyword_filter: Only inject images when vision-related keywords detected (default False)
            keywords: List of keywords that trigger image injection (default: common vision words)
            on_camera_state_change: Optional callback when camera state changes (receives bool)
        """
        super().__init__()
        self._context = context
        self._target_fps = target_fps
        self._frame_interval = 1.0 / target_fps
        self._last_frame_time = 0.0
        self._last_image = None
        self._inject_on_text = inject_on_text
        self._keyword_filter = keyword_filter
        self._on_camera_state_change = on_camera_state_change

        # Camera state tracking (for runtime system prompt updates)
        self._camera_active = False

        # Default vision-related keywords
        self._keywords = keywords or [
            'see', 'look', 'show', 'what', 'describe', 'image', 'picture',
            'video', 'color', 'object', 'room', 'view', 'watch', 'observe',
            'notice', 'appears', 'visible', 'display', 'scene', 'visual'
        ]

        # Phase 1: Image preprocessing configuration
        self._target_size = int(os.getenv("VISION_IMAGE_SIZE", "384"))
        self._image_quality = int(os.getenv("VISION_IMAGE_QUALITY", "85"))

        # Phase 2: Context pruning configuration
        self._max_images = int(os.getenv("VISION_MAX_IMAGES_IN_CONTEXT", "2"))
        self._injected_image_count = 0  # Track total number of injected images

        # Phase 3: Deduplication configuration
        self._enable_deduplication = os.getenv("VISION_ENABLE_DEDUPLICATION", "true").lower() in ("true", "1", "yes")
        self._last_injected_hash = None

        logger.info(f"[VisionContextInjector] Initialized (fps={target_fps}, inject_on_text={inject_on_text}, "
                   f"keyword_filter={keyword_filter}, keywords={len(self._keywords)} words, "
                   f"image_size={self._target_size}px, quality={self._image_quality}, "
                   f"max_images={self._max_images}, dedup={self._enable_deduplication}, "
                   f"camera_state_callback={'enabled' if on_camera_state_change else 'disabled'})")

    @property
    def camera_active(self) -> bool:
        """Check if camera is currently streaming frames."""
        return self._camera_active

    def _resize_image(self, image_data: bytes, size: tuple, format: str) -> tuple[bytes, tuple, str]:
        """
        Resize image for faster LLM processing.

        Args:
            image_data: Raw image bytes
            size: Original image dimensions (width, height)
            format: Image format (e.g., 'RGB', 'RGBA')

        Returns:
            Tuple of (resized_image_bytes, new_size, output_format)
        """
        try:
            # Skip resize if target size is -1 (disabled)
            if self._target_size <= 0:
                return image_data, size, format

            # Convert raw bytes to PIL Image
            img = Image.frombytes(format, size, image_data)

            # Convert RGBA to RGB (JPEG doesn't support transparency)
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                img = background
                output_format = 'RGB'
            else:
                output_format = format

            # Resize maintaining aspect ratio
            # thumbnail() modifies in-place and preserves aspect ratio
            img.thumbnail((self._target_size, self._target_size), Image.Resampling.LANCZOS)

            # Convert back to raw bytes for Pipecat
            # Pipecat will do JPEG compression when adding to context
            resized_data = img.tobytes()
            new_size = img.size

            # Resize stats calculated but not logged to reduce verbosity
            original_size = len(image_data)
            resized_size = len(resized_data)
            reduction = (1 - resized_size / original_size) * 100

            return resized_data, new_size, output_format

        except Exception as e:
            logger.error(f"[VisionContextInjector] Image resize failed: {e}")
            # Fallback to original image
            return image_data, size, format

    def _prune_old_images(self):
        """
        Remove old image messages from context when limit exceeded.
        Keeps only the N most recent images by dynamically searching for image content.
        """
        try:
            if self._injected_image_count <= self._max_images:
                return  # Within limit

            # Get current messages
            messages = list(self._context.get_messages())

            # Find all image messages by searching for image_url content
            image_indices = []
            for idx, msg in enumerate(messages):
                content = msg.get('content')
                # Check if content is a list with image_url type (OpenAI vision format)
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'image_url':
                            image_indices.append(idx)
                            break

            # Calculate how many to remove
            num_images_found = len(image_indices)
            if num_images_found <= self._max_images:
                logger.debug(f"[VisionContextInjector] Found {num_images_found} images, within limit of {self._max_images}")
                return

            num_to_remove = num_images_found - self._max_images
            indices_to_remove = image_indices[:num_to_remove]  # Remove oldest (first in list)

            # Remove old image messages (in reverse order to preserve indices)
            for idx in sorted(indices_to_remove, reverse=True):
                if idx < len(messages):
                    messages.pop(idx)
                    logger.debug(f"[VisionContextInjector] Removed old image message at index {idx}")

            # Update context
            self._context.set_messages(messages)

            logger.info(f"[VisionContextInjector] Pruned {num_to_remove} old images from context "
                       f"(found {num_images_found}, keeping {self._max_images} most recent)")

        except Exception as e:
            logger.error(f"[VisionContextInjector] Image pruning failed: {e}")

    def _should_inject_image(self, image_data: bytes) -> bool:
        """
        Check if image should be injected (deduplication).

        Args:
            image_data: Image bytes to check

        Returns:
            True if should inject, False if duplicate
        """
        if not self._enable_deduplication:
            return True

        # Hash the image data
        image_hash = hash(image_data)

        # Check if same as last injected
        if image_hash == self._last_injected_hash:
            logger.debug("[VisionContextInjector] Skipping duplicate image")
            return False

        # Update last injected hash
        self._last_injected_hash = image_hash
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Always call parent first
        await super().process_frame(frame, direction)

        # Handle video frames - store the latest frame
        if isinstance(frame, InputImageRawFrame):
            # Detect camera activation (first frame received)
            if not self._camera_active:
                self._camera_active = True
                logger.info("[VisionContextInjector] 📹 Camera activated - first frame received")

                # Notify callback about camera state change
                if self._on_camera_state_change:
                    try:
                        await self._on_camera_state_change(True)
                    except Exception as e:
                        logger.error(f"[VisionContextInjector] Camera state callback failed: {e}")

            current_time = asyncio.get_event_loop().time()
            if current_time - self._last_frame_time >= self._frame_interval:
                self._last_frame_time = current_time

                # Phase 1: Resize image before storing
                resized_image, new_size, output_format = self._resize_image(
                    frame.image, frame.size, frame.format
                )

                # Store the resized image
                self._last_image = {
                    'image': resized_image,
                    'size': new_size,
                    'format': output_format
                }
                # logger.debug(f"[VisionContextInjector] Stored frame: {new_size} (compressed)")
            # Don't push the frame downstream - we're handling it here
            return

        # When user sends text (from STT or text input), inject the latest image into context
        if isinstance(frame, (TextFrame, TranscriptionFrame)):
            text = getattr(frame, 'text', '')
            logger.debug(f"[VisionContextInjector] Received text frame: '{text[:50]}...' (has_image={self._last_image is not None})")

            # Check if we should inject image
            should_inject = False
            if self._last_image and self._inject_on_text:
                if self._keyword_filter:
                    # Check for vision-related keywords in text
                    text_lower = text.lower()
                    if any(keyword in text_lower for keyword in self._keywords):
                        should_inject = True
                        logger.info(f"[VisionContextInjector] Vision keyword detected in: '{text[:50]}...'")
                    else:
                        logger.debug(f"[VisionContextInjector] No vision keywords found, skipping image injection")
                else:
                    # No keyword filter, inject on all text
                    should_inject = True

            if should_inject:
                # Phase 3: Check deduplication before injecting
                if not self._should_inject_image(self._last_image['image']):
                    logger.debug("[VisionContextInjector] Skipping duplicate image injection")
                else:
                    logger.info(f"[VisionContextInjector] Injecting image into context with text: '{text[:50]}...'")
                    try:
                        # Inject the image
                        self._context.add_image_frame_message(
                            format=self._last_image['format'],
                            size=self._last_image['size'],
                            image=self._last_image['image'],
                            text=None  # Text is already in a separate frame
                        )

                        # Increment the injected image counter
                        self._injected_image_count += 1
                        logger.debug(f"[VisionContextInjector] Image injected (total: {self._injected_image_count})")

                        logger.info(f"[VisionContextInjector] ✓ Image added to LLM context")

                        # Phase 2: Prune old images if needed
                        self._prune_old_images()

                    except Exception as e:
                        logger.error(f"[VisionContextInjector] Failed to add image to context: {e}")

        # Pass through all other frames
        await self.push_frame(frame, direction)
