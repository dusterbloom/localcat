((.venv) ) peppi@MacBook server % clear && python core/bot.py
2025-09-16 15:17:50.466 | INFO     | pipecat:<module>:14 - ᓚᘏᗢ Pipecat 0.0.81 (Python 3.12.10 (main, Apr  8 2025, 11:35:47) [Clang 17.0.0 (clang-1700.0.13.3)]) ᓚᘏᗢ
2025-09-16 15:17:56.875 | INFO     | components.memory.config:<module>:248 - ⚙️ Configuration module initialized - centralized config management
2025-09-16 15:17:56.875 | INFO     | components.memory.config:<module>:249 - 🎯 Environment variables loaded and validated
2025-09-16 15:17:57.160 | INFO     | components.memory.memory_hotpath:<module>:83 - pycld3 not available, defaulting to English
2025-09-16 15:17:57.287 | INFO     | components.extraction.memory_extractor:<module>:447 - 🎯 MemoryExtractor initialized - dedicated extraction service
2025-09-16 15:17:57.287 | INFO     | components.extraction.memory_extractor:<module>:448 - 📊 Strategies: level3
2025-09-16 15:17:57.288 | INFO     | components.retrieval.memory_retriever:<module>:641 - 🎯 MemoryRetriever initialized - dedicated retrieval service
2025-09-16 15:17:57.288 | INFO     | components.retrieval.memory_retriever:<module>:642 - 📊 Features: MMR algorithm, entity expansion, LEANN integration, FTS fusion
2025-09-16 15:17:57.289 | INFO     | components.coreference.coreference_resolver:<module>:278 - 🎯 CoreferenceResolver initialized - dedicated coreference service
2025-09-16 15:17:57.289 | INFO     | components.coreference.coreference_resolver:<module>:279 - 📊 Features: Neural coreference, rule-based fallback, performance optimization
2025-09-16 15:17:57.290 | INFO     | components.extraction.assisted_extractor:<module>:374 - 🎯 AssistedExtractor initialized - dedicated LLM-assisted extraction service
2025-09-16 15:17:57.290 | INFO     | components.extraction.assisted_extractor:<module>:375 - 📊 Features: Classifier-based, JSON-based, fallback methods, performance optimization
2025-09-16 15:17:57.298 | INFO     | components.semantic.semantic_filter:<module>:375 - 🎯 SemanticRelationshipFilter initialized - advanced semantic relationship filtering
2025-09-16 15:17:57.298 | INFO     | components.semantic.semantic_filter:<module>:376 - 📊 Features: Semantic deduplication, generic relationship filtering, confidence scoring
2025-09-16 15:17:57.301 | INFO     | components.temporal.temporal_extractor:<module>:581 - 🎯 TemporalContextExtractor initialized - advanced temporal context extraction
2025-09-16 15:17:57.301 | INFO     | components.temporal.temporal_extractor:<module>:582 - 📊 Features: Time expression extraction, normalization, temporal relationship enhancement
2025-09-16 15:17:57.303 | INFO     | components.graph.graph_analyzer:<module>:371 - 🎯 KnowledgeGraphAnalyzer initialized - advanced graph analysis with NetworkX
2025-09-16 15:17:57.303 | INFO     | components.graph.graph_analyzer:<module>:372 - 📊 Features: Community detection, centrality analysis, graph statistics
2025-09-16 15:17:57.304 | INFO     | components.memory.hotmemory_facade:<module>:887 - 🎭 HotMemoryFacade initialized - backward compatibility maintained
2025-09-16 15:17:57.304 | INFO     | components.memory.hotmemory_facade:<module>:888 - 🔄 Using extracted services internally while preserving original interface
INFO:     Started server process [46082]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:7860 (Press CTRL+C to quit)
2025-09-16 15:18:13.427 | DEBUG    | pipecat.transports.network.webrtc_connection:_initialize:234 - Initializing new peer connection
2025-09-16 15:18:13.440 | DEBUG    | pipecat.transports.network.webrtc_connection:_create_answer:320 - Creating answer
2025-09-16 15:18:13.441 | DEBUG    | pipecat.transports.network.webrtc_connection:on_track:302 - Track audio received
2025-09-16 15:18:13.441 | DEBUG    | pipecat.transports.network.webrtc_connection:on_track:302 - Track video received
2025-09-16 15:18:13.444 | DEBUG    | pipecat.transports.network.webrtc_connection:on_icegatheringstatechange:298 - ICE gathering state is gathering
2025-09-16 15:18:13.503 | DEBUG    | pipecat.transports.network.webrtc_connection:on_icegatheringstatechange:298 - ICE gathering state is complete
2025-09-16 15:18:13.504 | DEBUG    | pipecat.transports.network.webrtc_connection:_create_answer:323 - Setting the answer after the local description is created
INFO:     127.0.0.1:49431 - "POST /api/offer HTTP/1.1" 200 OK
2025-09-16 15:18:13.504 | DEBUG    | pipecat.audio.vad.silero:__init__:147 - Loading Silero VAD model...
2025-09-16 15:18:13.611 | DEBUG    | pipecat.audio.vad.silero:__init__:169 - Loaded Silero VAD
2025-09-16 15:18:13.611 | DEBUG    | pipecat.audio.turn.smart_turn.local_smart_turn_v2:__init__:60 - Loading Local Smart Turn v2 model...
2025-09-16 15:18:16.397 | DEBUG    | pipecat.audio.turn.smart_turn.local_smart_turn_v2:__init__:74 - Loaded Local Smart Turn v2
2025-09-16 15:18:16.397 | INFO     | tts.tts_mlx_isolated:_get_worker_script_path:147 - Using worker script: /Users/peppi/Dev/localcat/server/tts/kokoro_worker.py
2025-09-16 15:18:16.473 | INFO     | components.processing.hotpath_processor:__init__:96 - HotMem storage: sqlite=/Users/peppi/Dev/localcat/server/components/data/memory.db lmdb=/Users/peppi/Dev/localcat/server/components/data/graph.lmdb
2025-09-16 15:18:16.474 | INFO     | components.retrieval.memory_retriever:__init__:82 - [MemoryRetriever] Initialized with LEANN=✓, fusion=✓
2025-09-16 15:18:16.474 | INFO     | components.coreference.coreference_resolver:__init__:63 - [CoreferenceResolver] Initialized with mode=lite neural=✗, max_entities=24
2025-09-16 15:18:16.474 | INFO     | components.extraction.assisted_extractor:__init__:59 - [AssistedExtractor] Initialized with model=google/gemma-3-270m, enabled=✗
2025-09-16 15:18:16.475 | INFO     | components.semantic.semantic_filter:__init__:69 - [SemanticRelationshipFilter] Initialized with enabled=✓
2025-09-16 15:18:16.475 | INFO     | components.temporal.temporal_extractor:__init__:100 - [TemporalContextExtractor] Initialized with enabled=✓
2025-09-16 15:18:16.475 | INFO     | components.graph.graph_analyzer:__init__:70 - [KnowledgeGraphAnalyzer] Initialized with enabled=✓, louvain=✓
2025-09-16 15:18:16.478 | INFO     | components.session.session_store:_init_database:115 - 🗃️ SessionStore initialized at sessions.db
2025-09-16 15:18:16.478 | INFO     | components.memory.config:log_configuration:235 - 🔧 HotMemory Configuration:
2025-09-16 15:18:16.479 | INFO     | components.memory.config:log_configuration:236 -   📊 Feature flags: FeatureFlags(use_srl=False, use_onnx_ner=False, use_onnx_srl=False, use_coref=True, use_dspy=False, use_gliner=False, use_leann=True, retrieval_fusion=True, assisted_enabled=False, use_semantic_filter=True, use_temporal_extraction=True, use_graph_analysis=True, session_context_enabled=True, session_navigation_enabled=True, temporal_awareness_enabled=True)
2025-09-16 15:18:16.479 | INFO     | components.memory.config:log_configuration:237 -   🤖 Assisted model: google/gemma-3-270m
2025-09-16 15:18:16.479 | INFO     | components.memory.config:log_configuration:238 -   🧠 LEANN: enabled=True, complexity=16
2025-09-16 15:18:16.479 | INFO     | components.memory.config:log_configuration:239 -   ⚡ Cache size: 1000
2025-09-16 15:18:16.479 | INFO     | components.memory.config:log_configuration:240 -   🎯 Confidence threshold: 0.3
/Users/peppi/Dev/localcat/server/.venv/lib/python3.12/site-packages/spacy/util.py:922: UserWarning: [W095] Model 'en_core_web_trf' (3.7.3) was trained with spaCy v3.7.2 and may not be 100% compatible with the current version (3.8.7). If you see errors or degraded performance, download a newer compatible model or retrain your custom model with the current spaCy version. For more details and available updates, run: python -m spacy validate
  warnings.warn(warn_msg)
2025-09-16 15:18:18.414 | INFO     | components.memory.hotmemory_facade:rebuild_from_store:873 - [HotMem] Rebuilt indices from store: entities=8, edges=5
2025-09-16 15:18:18.415 | INFO     | components.processing.hotpath_processor:__init__:149 - HotPathMemoryProcessor initialized for user: peppi
2025-09-16 15:18:18.415 | INFO     | components.monitoring.metrics_collector:_init_database:118 - Metrics database initialized at /Users/peppi/Dev/localcat/data/metrics.db
2025-09-16 15:18:18.417 | DEBUG    | components.monitoring.metrics_collector:register_metric:251 - Registered metric: system.cpu_percent
2025-09-16 15:18:18.417 | DEBUG    | components.monitoring.metrics_collector:register_metric:251 - Registered metric: system.memory_percent
2025-09-16 15:18:18.419 | DEBUG    | components.monitoring.metrics_collector:register_metric:251 - Registered metric: system.memory_available_gb
2025-09-16 15:18:18.419 | DEBUG    | components.monitoring.metrics_collector:register_metric:251 - Registered metric: system.disk_percent
2025-09-16 15:18:18.420 | DEBUG    | components.monitoring.metrics_collector:register_metric:251 - Registered metric: app.memory_entities
2025-09-16 15:18:18.421 | DEBUG    | components.monitoring.metrics_collector:register_metric:251 - Registered metric: app.memory_edges
2025-09-16 15:18:18.422 | DEBUG    | components.monitoring.metrics_collector:register_metric:251 - Registered metric: app.memory_mentions
2025-09-16 15:18:18.422 | DEBUG    | components.monitoring.metrics_collector:register_metric:251 - Registered metric: perf.extraction_latency
2025-09-16 15:18:18.423 | DEBUG    | components.monitoring.metrics_collector:register_metric:251 - Registered metric: perf.retrieval_latency
2025-09-16 15:18:18.424 | DEBUG    | components.monitoring.metrics_collector:register_metric:251 - Registered metric: perf.tts_latency
2025-09-16 15:18:18.424 | DEBUG    | components.monitoring.metrics_collector:register_collector:256 - Registered collector: system_metrics
2025-09-16 15:18:18.424 | DEBUG    | components.monitoring.metrics_collector:register_collector:256 - Registered collector: memory_metrics
2025-09-16 15:18:18.424 | INFO     | components.monitoring.metrics_collector:start_collection:320 - Started metrics collection
2025-09-16 15:18:18.424 | INFO     | components.monitoring.health_monitor:add_service:161 - Added service to monitoring: ollama
2025-09-16 15:18:18.424 | INFO     | components.monitoring.health_monitor:add_service:161 - Added service to monitoring: lm_studio
2025-09-16 15:18:18.424 | INFO     | components.monitoring.health_monitor:add_service:161 - Added service to monitoring: database
2025-09-16 15:18:18.425 | INFO     | components.monitoring.health_monitor:add_service:161 - Added service to monitoring: leann_index
2025-09-16 15:18:18.425 | INFO     | components.monitoring.health_monitor:start_monitoring:180 - Started health monitoring for 4 services
2025-09-16 15:18:18.427 | INFO     | components.monitoring.alerting_system:_init_database:184 - Alerting database initialized at /Users/peppi/Dev/localcat/data/alerts.db
2025-09-16 15:18:18.429 | INFO     | components.monitoring.alerting_system:add_rule:353 - Added alert rule: high_cpu_usage
2025-09-16 15:18:18.430 | INFO     | components.monitoring.alerting_system:add_rule:353 - Added alert rule: critical_cpu_usage
2025-09-16 15:18:18.431 | INFO     | components.monitoring.alerting_system:add_rule:353 - Added alert rule: high_memory_usage
2025-09-16 15:18:18.432 | INFO     | components.monitoring.alerting_system:add_rule:353 - Added alert rule: low_disk_space
2025-09-16 15:18:18.433 | INFO     | components.monitoring.alerting_system:add_rule:353 - Added alert rule: service_unhealthy
2025-09-16 15:18:18.434 | INFO     | components.monitoring.alerting_system:add_rule:353 - Added alert rule: slow_extraction
2025-09-16 15:18:18.435 | INFO     | components.monitoring.alerting_system:add_rule:353 - Added alert rule: slow_tts
2025-09-16 15:18:18.436 | INFO     | components.monitoring.alerting_system:add_channel:374 - Added notification channel: console
2025-09-16 15:18:18.437 | INFO     | components.monitoring.alerting_system:add_channel:374 - Added notification channel: log
2025-09-16 15:18:18.437 | INFO     | components.monitoring.alerting_system:start_monitoring:385 - Started alert monitoring
2025-09-16 15:18:18.437 | INFO     | __main__:run_bot:329 - ✅ Monitoring system initialized successfully
2025-09-16 15:18:18.437 | DEBUG    | pipecat.processors.frame_processor:link:499 - Linking Pipeline#0::Source -> SmallWebRTCInputTransport#0
2025-09-16 15:18:18.438 | DEBUG    | pipecat.processors.frame_processor:link:499 - Linking SmallWebRTCInputTransport#0 -> WhisperSTTServiceMLX#0
2025-09-16 15:18:18.438 | DEBUG    | pipecat.processors.frame_processor:link:499 - Linking WhisperSTTServiceMLX#0 -> RTVIProcessor#0
2025-09-16 15:18:18.438 | DEBUG    | pipecat.processors.frame_processor:link:499 - Linking RTVIProcessor#0 -> HotPathMemoryProcessor#0
2025-09-16 15:18:18.438 | DEBUG    | pipecat.processors.frame_processor:link:499 - Linking HotPathMemoryProcessor#0 -> OpenAIUserContextAggregator#0
2025-09-16 15:18:18.438 | DEBUG    | pipecat.processors.frame_processor:link:499 - Linking OpenAIUserContextAggregator#0 -> OpenAILLMService#0
2025-09-16 15:18:18.438 | DEBUG    | pipecat.processors.frame_processor:link:499 - Linking OpenAILLMService#0 -> TTSMLXIsolated#0
2025-09-16 15:18:18.439 | DEBUG    | pipecat.processors.frame_processor:link:499 - Linking TTSMLXIsolated#0 -> SmallWebRTCOutputTransport#0
2025-09-16 15:18:18.439 | DEBUG    | pipecat.processors.frame_processor:link:499 - Linking SmallWebRTCOutputTransport#0 -> OpenAIAssistantContextAggregator#0
2025-09-16 15:18:18.439 | DEBUG    | pipecat.processors.frame_processor:link:499 - Linking OpenAIAssistantContextAggregator#0 -> Pipeline#0::Sink
2025-09-16 15:18:18.440 | DEBUG    | pipecat.processors.frame_processor:link:499 - Linking PipelineTask#0::Source -> Pipeline#0
2025-09-16 15:18:18.440 | DEBUG    | pipecat.processors.frame_processor:link:499 - Linking Pipeline#0 -> PipelineTask#0::Sink
2025-09-16 15:18:18.440 | WARNING  | pipecat.utils.base_object:add_event_handler:108 - Event handler on_first_participant_joined not registered
2025-09-16 15:18:18.440 | WARNING  | pipecat.utils.base_object:add_event_handler:108 - Event handler on_participant_left not registered
2025-09-16 15:18:18.440 | DEBUG    | pipecat.pipeline.runner:run:71 - Runner PipelineRunner#0 started running PipelineTask#0
09/16/2025 15:18:18 - INFO -     Connection(0) Check CandidatePair(('192.168.1.4', 61401) -> ('192.168.1.4', 49256)) State.FROZEN -> State.WAITING
09/16/2025 15:18:18 - INFO -     Connection(0) Check CandidatePair(('192.168.1.4', 61401) -> ('82.59.32.250', 49256)) State.FROZEN -> State.WAITING
2025-09-16 15:18:18.441 | INFO     | services.summarizer:periodic_summarizer:155 - [Summarizer] Enabled (model=qwen/qwen3-4b, base=http://127.0.0.1:1234/v1, interval=300s)
2025-09-16 15:18:18.449 | ERROR    | pipecat.processors.frame_processor:_check_started:730 - RTVIProcessor#0 Trying to process TransportMessageUrgentFrame#0(message: {'label': 'rtvi-ai', 'type': 'metrics', 'data': {'ttfb': [{'processor': 'WhisperSTTServiceMLX#0', 'value': 0.0}, {'processor': 'OpenAILLMService#0', 'value': 0.0}, {'processor': 'TTSMLXIsolated#0', 'value': 0.0}], 'processing': [{'processor': 'WhisperSTTServiceMLX#0', 'value': 0.0}, {'processor': 'OpenAILLMService#0', 'value': 0.0}, {'processor': 'TTSMLXIsolated#0', 'value': 0.0}]}}) but StartFrame not received yet
2025-09-16 15:18:18.450 | DEBUG    | pipecat.transports.network.webrtc_connection:on_iceconnectionstatechange:292 - ICE connection state is checking, connection is connecting
2025-09-16 15:18:18.450 | DEBUG    | pipecat.transports.network.webrtc_connection:_handle_new_connection_state:484 - Connection state changed to: connecting
09/16/2025 15:18:18 - INFO -     Connection(0) Check CandidatePair(('192.168.1.4', 61401) -> ('192.168.1.4', 49256)) State.WAITING -> State.IN_PROGRESS
2025-09-16 15:18:18.457 | DEBUG    | pipecat.audio.vad.vad_analyzer:set_params:150 - Setting VAD params to: confidence=0.7 start_secs=0.2 stop_secs=0.2 min_volume=0.6
2025-09-16 15:18:18.458 | INFO     | pipecat.transports.network.small_webrtc:connect:420 - Connecting to Small WebRTC
09/16/2025 15:18:18 - INFO -     server listening on [::1]:9090
09/16/2025 15:18:18 - INFO -     server listening on 127.0.0.1:9090
2025-09-16 15:18:18.459 | DEBUG    | pipecat_whisker.observer:_start_task_handler:177 - ᓚᘏᗢ Whisker running at ws://localhost:9090
09/16/2025 15:18:18 - INFO -     Connection(0) Check CandidatePair(('192.168.1.4', 61401) -> ('192.168.1.4', 49256)) State.IN_PROGRESS -> State.SUCCEEDED
09/16/2025 15:18:18 - INFO -     Connection(0) Check CandidatePair(('192.168.1.4', 61401) -> ('82.59.32.250', 49256)) State.WAITING -> State.FAILED
09/16/2025 15:18:18 - INFO -     Connection(0) ICE completed
2025-09-16 15:18:18.462 | INFO     | pipecat.transports.network.small_webrtc:connect:420 - Connecting to Small WebRTC
2025-09-16 15:18:18.490 | DEBUG    | pipecat.transports.network.webrtc_connection:on_iceconnectionstatechange:292 - ICE connection state is completed, connection is connecting
2025-09-16 15:18:18.495 | DEBUG    | pipecat.transports.network.webrtc_connection:_handle_new_connection_state:484 - Connection state changed to: connected
2025-09-16 15:18:18.496 | DEBUG    | pipecat.transports.network.small_webrtc:on_connected:241 - Peer connection established.
2025-09-16 15:18:18.496 | WARNING  | pipecat.transports.network.webrtc_connection:screen_video_input_track:569 - No screen video transceiver is available
2025-09-16 15:18:18.496 | DEBUG    | pipecat.transports.network.webrtc_connection:replace_audio_track:407 - Replacing audio track audio
2025-09-16 15:18:18.497 | DEBUG    | pipecat.transports.network.webrtc_connection:_handle_signalling_message:602 - Signalling message received: {'type': 'trackStatus', 'receiver_index': 0, 'enabled': True}
2025-09-16 15:18:18.498 | DEBUG    | pipecat.transports.network.webrtc_connection:_handle_signalling_message:602 - Signalling message received: {'type': 'trackStatus', 'receiver_index': 1, 'enabled': False}
2025-09-16 15:18:18.498 | DEBUG    | pipecat.transports.network.small_webrtc:push_app_message:652 - Received app message inside SmallWebRTCInputTransport  {'label': 'rtvi-ai', 'type': 'client-ready', 'data': {'version': '1.0.0', 'about': {'library': '@pipecat-ai/client-react', 'library_version': '1.0.1', 'platform_details': {'browser': 'Chrome', 'browser_version': '140.0.0.0', 'platform_type': 'desktop', 'engine': 'Blink'}, 'platform': 'macOS', 'platform_version': '10.15.7'}}, 'id': 'f3d46413'}
2025-09-16 15:18:18.498 | DEBUG    | pipecat.processors.frameworks.rtvi:_handle_client_ready:1427 - Received client-ready: version 1.0.0
2025-09-16 15:18:18.499 | DEBUG    | pipecat.processors.frameworks.rtvi:_handle_client_ready:1437 - Client Details: library='@pipecat-ai/client-react' library_version='1.0.1' platform='macOS' platform_version='10.15.7' platform_details={'browser': 'Chrome', 'browser_version': '140.0.0.0', 'platform_type': 'desktop', 'engine': 'Blink'}
/Users/peppi/Dev/localcat/server/.venv/lib/python3.12/site-packages/pipecat/processors/frameworks/rtvi.py:1545: DeprecationWarning: Configuration helpers are deprecated. If your application needs this behavior, use custom server and client messages.
  warnings.warn(
2025-09-16 15:18:18.520 | DEBUG    | tts.tts_mlx_isolated:run_tts:274 - TTSMLXIsolated#0: Generating TTS [Hello!]
2025-09-16 15:18:18.520 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 6
2025-09-16 15:18:18.521 | DEBUG    | tts.tts_mlx_isolated:_send_command:178 - Starting worker process...
2025-09-16 15:18:18.555 | INFO     | tts.tts_mlx_isolated:_start_worker:168 - Started mlx-community/Kokoro-82M-bf16 worker process: 46152
2025-09-16 15:18:18.555 | DEBUG    | tts.tts_mlx_isolated:_send_command:184 - Sending command: {'cmd': 'init', 'model': 'mlx-community/Kokoro-82M-bf16', 'voice': 'af_heart'}
2025-09-16 15:18:18.943 | DEBUG    | components.session.session_store:add_message:156 - 💬 Added assistant message to session session_1758028698_peppi
2025-09-16 15:18:18.943 | DEBUG    | components.processing.hotpath_processor:store_assistant_response:235 - 💾 Stored assistant response for session session_1758028698_peppi
2025-09-16 15:18:19.790 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:348 - User started speaking
2025-09-16 15:18:19.793 | DEBUG    | tts.tts_mlx_isolated:run_tts:313 - TTSMLXIsolated#0: Finished TTS [Hello!]
2025-09-16 15:18:19.793 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 1.2726020812988281
2025-09-16 15:18:19.793 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 1.274143934249878
2025-09-16 15:18:20.424 | DEBUG    | pipecat.audio.turn.smart_turn.base_smart_turn:analyze_end_of_turn:157 - End of Turn result: EndOfTurnState.COMPLETE
2025-09-16 15:18:20.425 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:372 - User stopped speaking
Fetching 4 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 75234.15it/s]
2025-09-16 15:18:22.967 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - WhisperSTTServiceMLX#0 TTFB: 2.2541589736938477
2025-09-16 15:18:22.967 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - WhisperSTTServiceMLX#0 processing time: 2.254439353942871
2025-09-16 15:18:26.631 | DEBUG    | tts.tts_mlx_isolated:_send_command:213 - Worker response: {"success": true}
2025-09-16 15:18:27.255 | DEBUG    | pipecat.transports.network.webrtc_connection:_handle_signalling_message:602 - Signalling message received: {'type': 'trackStatus', 'receiver_index': 0, 'enabled': False}
2025-09-16 15:18:41.688 | DEBUG    | pipecat.transports.network.webrtc_connection:_handle_signalling_message:602 - Signalling message received: {'type': 'trackStatus', 'receiver_index': 0, 'enabled': True}
2025-09-16 15:18:44.031 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:348 - User started speaking
2025-09-16 15:18:44.751 | DEBUG    | pipecat.audio.turn.smart_turn.base_smart_turn:analyze_end_of_turn:157 - End of Turn result: EndOfTurnState.COMPLETE
2025-09-16 15:18:44.752 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:372 - User stopped speaking
2025-09-16 15:18:45.633 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - WhisperSTTServiceMLX#0 TTFB: 0.8797850608825684
2025-09-16 15:18:45.634 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - WhisperSTTServiceMLX#0 processing time: 0.8802409172058105
2025-09-16 15:18:45.634 | DEBUG    | pipecat.services.whisper.stt:run_stt:511 - Transcription: [ Thank you, officers. ]
2025-09-16 15:18:45.639 | INFO     | components.processing.hotpath_processor:process_frame:466 - [HotMem] TranscriptionFrame received: is_final=None text_len=22 text=' Thank you, officers. '
2025-09-16 15:18:45.639 | INFO     | components.session.session_store:create_session:133 - 📝 Created session: session_1758028698_peppi for user: peppi
2025-09-16 15:18:45.640 | DEBUG    | components.processing.hotpath_processor:_ensure_session:227 - 📝 Session ensured: session_1758028698_peppi for user: peppi
2025-09-16 15:18:45.640 | INFO     | components.processing.hotpath_processor:process_frame:485 - [HotMem] Processing transcription (is_final=None): ' Thank you, officers. '
2025-09-16 15:18:45.983 | DEBUG    | components.session.session_store:add_message:156 - 💬 Added user message to session session_1758028698_peppi
2025-09-16 15:18:45.986 | INFO     | components.memory.memory_intent:__init__:365 - ⚡ Using Enhanced Rule V2 classifier (100% accuracy, <1ms)
2025-09-16 15:18:46.515 | DEBUG    | components.memory.hotmemory_facade:process_turn:251 - Skipping extraction for pure_question:  Thank you, officers. ...
2025-09-16 15:18:46.589 | DEBUG    | components.retrieval.memory_retriever:retrieve_context:93 - [MemoryRetriever] Retrieval context: query=' Than...', entities=['Thank'], total_edges=10
2025-09-16 15:18:46.592 | INFO     | components.processing.hotpath_processor:_process_transcription:533 - [HotMem] Prepared 5 memory bullets for injection
2025-09-16 15:18:46.592 | INFO     | components.processing.hotpath_processor:_process_transcription:550 - [HotMem] Summary: saved=0, pending_bullets=6, turn=1
2025-09-16 15:18:46.592 | INFO     | components.processing.hotpath_processor:_inject_memory_context:596 - [HotMem] Packing context with 5 bullets
2025-09-16 15:18:46.595 | INFO     | components.processing.hotpath_processor:_inject_memory_context:638 - [HotMem] Pack stats: total=377 sys=139 mem=214 sum=17 dlg=16
2025-09-16 15:18:46.603 | DEBUG    | pipecat.transports.network.webrtc_connection:_handle_signalling_message:602 - Signalling message received: {'type': 'trackStatus', 'receiver_index': 0, 'enabled': False}
2025-09-16 15:18:46.654 | DEBUG    | pipecat.services.openai.base_llm:_stream_chat_completions:247 - OpenAILLMService#0: Generating chat [[{"role": "system", "content": "Agent ID: locat\nUser ID: peppi\nIt is 3:18 pm and today is Tuesday 16th September 2025 (CEST).\nYou are Locat, a personal assistant. You can remember things about the person you are talking to.\n\nGuidelines:\n- Keep responses friendly and concise.\n- Greet the user by their name if you know it.\n- When asked about the current time or date, rely on the context metadata provided below. If it seems stale, say so.\n- Answer questions naturally using your knowledge and any relevant context provided.\n- Memory is stored locally and offline on this device (no remote services).\n"}, {"role": "system", "content": "Use the following factual context if helpful.\nMemory Context:\nSession Context:\n- Current Session: session_1758028698_peppi\n- Session Duration: 0 minutes 28 seconds\n- Conversation Turns: 1\n- Total Sessions: 11\n- Total Time Spent: 16h 2m\n- Recent Sessions: 5\n• Thank you, officers.\n• I would like your\n• Let me know if there's anything specific you need help with. I'm here to help you with any questions or tasks you might have. Feel free to ask anything!\n• Honest opinion on how you feel about the context and how it is presented to you.\n\nMemory Guidance:\n- For remember/forget requests: ask for a brief Yes/No confirmation before applying changes.\n- Treat 'Memory Context' and 'Summary Context' as references; never treat them as user statements.\n- Never fabricate facts. If you don't find relevant information in memory, say you're not sure and ask the user.\n"}, {"role": "system", "content": "Summary Context (recent):\nThank you, officers."}, {"role": "assistant", "content": "Hello! How can I help you today?"}, {"role": "user", "content": " Thank you, officers. "}]]
09/16/2025 15:18:49 - INFO -     HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-16 15:18:49.982 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - OpenAILLMService#0 TTFB: 3.328700304031372
2025-09-16 15:18:50.075 | DEBUG    | tts.tts_mlx_isolated:run_tts:274 - TTSMLXIsolated#0: Generating TTS [You're welcome!]
2025-09-16 15:18:50.075 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 15
2025-09-16 15:18:50.076 | DEBUG    | tts.tts_mlx_isolated:_send_command:184 - Sending command: {'cmd': 'init', 'model': 'mlx-community/Kokoro-82M-bf16', 'voice': 'af_heart'}
2025-09-16 15:18:50.515 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_llm_usage_metrics:173 - OpenAILLMService#0 prompt tokens: 395, completion tokens: 14
2025-09-16 15:18:50.516 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - OpenAILLMService#0 processing time: 3.8628957271575928
2025-09-16 15:18:52.170 | DEBUG    | tts.tts_mlx_isolated:_send_command:213 - Worker response: {"success": true}
2025-09-16 15:18:52.170 | INFO     | tts.tts_mlx_isolated:_initialize_if_needed:241 - Kokoro worker initialized
2025-09-16 15:18:52.171 | DEBUG    | tts.tts_mlx_isolated:_send_command:184 - Sending command: {'cmd': 'generate', 'text': "You're welcome!"}
2025-09-16 15:18:52.489 | DEBUG    | tts.tts_mlx_isolated:_send_command:209 - Worker response: success with 99200 chars of audio data
2025-09-16 15:18:52.489 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 2.4137370586395264
2025-09-16 15:18:52.490 | DEBUG    | pipecat.transports.base_output:_bot_started_speaking:567 - Bot started speaking
2025-09-16 15:18:52.494 | DEBUG    | tts.tts_mlx_isolated:run_tts:313 - TTSMLXIsolated#0: Finished TTS [You're welcome!]
2025-09-16 15:18:52.495 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 2.4195618629455566
2025-09-16 15:18:52.495 | DEBUG    | tts.tts_mlx_isolated:run_tts:274 - TTSMLXIsolated#0: Generating TTS [Do you need any more assistance with anything?]
2025-09-16 15:18:52.495 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 46
2025-09-16 15:18:52.495 | DEBUG    | tts.tts_mlx_isolated:_send_command:184 - Sending command: {'cmd': 'generate', 'text': 'Do you need any more assistance with anything?'}
2025-09-16 15:18:52.928 | DEBUG    | tts.tts_mlx_isolated:_send_command:209 - Worker response: success with 179200 chars of audio data
2025-09-16 15:18:52.929 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 0.43391895294189453
2025-09-16 15:18:52.934 | DEBUG    | tts.tts_mlx_isolated:run_tts:313 - TTSMLXIsolated#0: Finished TTS [Do you need any more assistance with anything?]
2025-09-16 15:18:52.934 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 0.4396851062774658
2025-09-16 15:18:57.104 | DEBUG    | components.session.session_store:add_message:156 - 💬 Added assistant message to session session_1758028698_peppi
2025-09-16 15:18:57.105 | DEBUG    | components.processing.hotpath_processor:store_assistant_response:235 - 💾 Stored assistant response for session session_1758028698_peppi
2025-09-16 15:18:57.158 | DEBUG    | pipecat.transports.base_output:_bot_stopped_speaking:583 - Bot stopped speaking
2025-09-16 15:23:24.790 | INFO     | services.summarizer:periodic_summarizer:200 - [Summarizer] Stored periodic session summary to FTS
2025-09-16 15:23:24.794 | DEBUG    | components.session.session_store:add_message:156 - 💬 Added assistant message to session session_1758028698_peppi
2025-09-16 15:23:24.794 | DEBUG    | components.processing.hotpath_processor:store_assistant_response:235 - 💾 Stored assistant response for session session_1758028698_peppi
2025-09-16 15:23:56.691 | WARNING  | pipecat.pipeline.task:_idle_timeout_detected:726 - Idle timeout detected. Last 10 frames received:
2025-09-16 15:23:56.691 | WARNING  | pipecat.pipeline.task:_idle_timeout_detected:728 - Frame 1: TTSAudioRawFrame#113(pts: None, destination: None, size: 1920, frames: 960, sample_rate: 24000, channels: 1)
2025-09-16 15:23:56.691 | WARNING  | pipecat.pipeline.task:_idle_timeout_detected:728 - Frame 2: TTSAudioRawFrame#114(pts: None, destination: None, size: 1920, frames: 960, sample_rate: 24000, channels: 1)
2025-09-16 15:23:56.691 | WARNING  | pipecat.pipeline.task:_idle_timeout_detected:728 - Frame 3: BotSpeakingFrame#42
2025-09-16 15:23:56.691 | WARNING  | pipecat.pipeline.task:_idle_timeout_detected:728 - Frame 4: TTSAudioRawFrame#115(pts: None, destination: None, size: 1920, frames: 960, sample_rate: 24000, channels: 1)
2025-09-16 15:23:56.691 | WARNING  | pipecat.pipeline.task:_idle_timeout_detected:728 - Frame 5: TTSAudioRawFrame#116(pts: None, destination: None, size: 1920, frames: 960, sample_rate: 24000, channels: 1)
2025-09-16 15:23:56.691 | WARNING  | pipecat.pipeline.task:_idle_timeout_detected:728 - Frame 6: TTSAudioRawFrame#117(pts: None, destination: None, size: 1920, frames: 960, sample_rate: 24000, channels: 1)
2025-09-16 15:23:56.691 | WARNING  | pipecat.pipeline.task:_idle_timeout_detected:728 - Frame 7: TTSStoppedFrame#2
2025-09-16 15:23:56.692 | WARNING  | pipecat.pipeline.task:_idle_timeout_detected:728 - Frame 8: OpenAILLMContextFrame#1
2025-09-16 15:23:56.692 | WARNING  | pipecat.pipeline.task:_idle_timeout_detected:728 - Frame 9: OpenAILLMContextAssistantTimestampFrame#0
2025-09-16 15:23:56.692 | WARNING  | pipecat.pipeline.task:_idle_timeout_detected:728 - Frame 10: BotStoppedSpeakingFrame#0
2025-09-16 15:23:56.692 | WARNING  | pipecat.pipeline.task:_idle_timeout_detected:732 - Idle pipeline detected, cancelling pipeline task...
2025-09-16 15:23:56.692 | DEBUG    | pipecat.pipeline.task:_cancel:428 - Cancelling pipeline task PipelineTask#0
2025-09-16 15:23:56.694 | INFO     | pipecat.transports.network.small_webrtc:disconnect:430 - Disconnecting to Small WebRTC
2025-09-16 15:23:56.696 | DEBUG    | pipecat.transports.network.webrtc_connection:on_iceconnectionstatechange:292 - ICE connection state is closed, connection is closed
2025-09-16 15:23:56.696 | DEBUG    | pipecat.transports.network.webrtc_connection:_handle_new_connection_state:484 - Connection state changed to: closed
2025-09-16 15:23:56.697 | INFO     | __main__:handle_disconnected:503 - Discarding peer connection for pc_id: SmallWebRTCConnection#0
2025-09-16 15:23:56.697 | DEBUG    | pipecat.transports.network.small_webrtc:on_closed:251 - Client connection closed.
2025-09-16 15:23:56.698 | DEBUG    | pipecat.pipeline.task:run:397 - Pipeline task PipelineTask#0 has finished, cleaning up resources
2025-09-16 15:23:56.698 | INFO     | components.processing.hotpath_processor:cleanup:780 - HotPathMemoryProcessor cleanup complete
2025-09-16 15:23:56.699 | WARNING  | pipecat.pipeline.task:_print_dangling_tasks:741 - Dangling tasks detected: ['HotPathMemoryProcessor#0::__input_frame_task_handler']
2025-09-16 15:23:56.699 | DEBUG    | pipecat.pipeline.runner:run:91 - Runner PipelineRunner#0 finished running PipelineTask#0
2025-09-16 15:23:56.699 | INFO     | services.summarizer:periodic_summarizer:227 - [Summarizer] Cancelled
2025-09-16 15:23:56.700 | INFO     | components.monitoring.health_monitor:stop_monitoring:202 - Stopped health monitoring
2025-09-16 15:23:56.700 | INFO     | components.monitoring.metrics_collector:stop_collection:336 - Stopped metrics collection
2025-09-16 15:23:56.700 | INFO     | components.monitoring.alerting_system:stop_monitoring:401 - Stopped alert monitoring
2025-09-16 15:23:56.700 | INFO     | __main__:run_bot:479 - Rebuilding LEANN index (finalizer) with 5 docs at ../data/memory_vectors.leann
09/16/2025 15:23:57 - INFO -     Load pretrained SentenceTransformer: facebook/contriever
09/16/2025 15:23:57 - WARNING -          No sentence-transformers model found with name facebook/contriever. Creating a new one with mean pooling.
Writing passages: 100%|█| 5/5 [00:00<00:00, 2
Batches: 100%|█| 1/1 [00:00<00:00, 17.39it/s]
09/16/2025 15:23:58 - WARNING -          Converting data to float32, shape: (5, 768)
M: 64 for level: 0
09/16/2025 15:23:58 - INFO -     INFO: Converting HNSW index to CSR-pruned format...
Starting conversion: ../data/memory_vectors.index -> ../data/memory_vectors.csr.tmp
[0.00s] Reading Index HNSW header...
[0.00s]   Header read: d=768, ntotal=5
[0.00s] Reading HNSW struct vectors...
  Reading vector (dtype=<class 'numpy.float64'>, fmt='d')... Count=6, Bytes=48
[0.00s]   Read assign_probas (6)
  Reading vector (dtype=<class 'numpy.int32'>, fmt='i')... Count=7, Bytes=28
[0.21s]   Read cum_nneighbor_per_level (7)
  Reading vector (dtype=<class 'numpy.int32'>, fmt='i')... Count=5, Bytes=20
[0.43s]   Read levels (5)
[0.62s]   Probing for compact storage flag...
[0.62s]   Found compact flag: False
[0.62s]   Compact flag is False, reading original format...
[0.62s]   Probing for potential extra byte before non-compact offsets...
[0.62s]   Found and consumed an unexpected 0x00 byte.
  Reading vector (dtype=<class 'numpy.uint64'>, fmt='Q')... Count=6, Bytes=48
[0.62s]   Read offsets (6)
[0.82s]   Attempting to read neighbors vector...
  Reading vector (dtype=<class 'numpy.int32'>, fmt='i')... Count=320, Bytes=1280
[0.82s]   Read neighbors (320)
[1.01s]   Read scalar params (ep=4, max_lvl=0)
[1.01s] Checking for storage data...
[1.01s]   Found storage fourcc: 49467849.
[1.01s] Converting to CSR format...
[1.01s]   Conversion loop finished.                        
[1.01s] Running validation checks...
    Checking total valid neighbor count...
    OK: Total valid neighbors = 20
    Checking final pointer indices...
    OK: Final pointers match data size.
[1.01s] Deleting original neighbors and offsets arrays...
    CSR Stats: |data|=20, |level_ptr|=10
[1.19s] Writing CSR HNSW graph data in FAISS-compatible order...
   Pruning embeddings: Writing NULL storage marker.
[1.38s] Conversion complete.
09/16/2025 15:23:59 - INFO -     ✅ CSR conversion successful.
09/16/2025 15:23:59 - INFO -     INFO: Replaced original index with CSR-pruned version at '../data/memory_vectors.index'
2025-09-16 15:23:59.963 | INFO     | services.leann_adapter:rebuild_leann_index:31 - LEANN index rebuilt at ../data/memory_vectors.leann
