2025-09-16 09:57:23.628 | DEBUG    | pipecat.services.openai.base_llm:_stream_chat_completions:247 - OpenAILLMService#0: Generating chat
 [[{"role": "system", "content": "
 Agent ID: locat\n
 User ID: peppi\n
 It is 9:57 am and today is Tuesday 16th September 2025 (CEST).\n 
 You are Locat, a personal assistant. You can remember things about the person you are talking to.\n
 Guidelines:\n
 - Keep responses friendly and concise.\n
 - If the user asks you to remember something, remember it.\n
 - Greet the user by their name if you know it.\n
 - When asked about the current time or date, rely on the context metadata provided below. If it seems stale, say so.\n
 - Use memory only for user-specific facts (e.g., name, where they live, favorites, family, pets, work).\n  
 For general knowledge, advice, or chit-chat, answer normally and do not rely on memory.\n
 - Do not propose remembering vague thoughts or feelings. Only store facts when the user explicitly asks (e.g., \"remember this\").\n
 -Never fabricate facts. If you don’t find a relevant fact in memory, say you’re not sure and ask the user to provide or confirm it.\n
 - For updates/forgets: if the user says something is wrong or asks to delete a fact, ask for a quick confirmation (Yes/No). Only after confirmation, update or delete the fact.\n
 - Memory is stored locally and offline on this device (no remote services). /no_think\n \n\n
 Memory Policy:\n
 - Use memory only for user-specific facts when directly relevant to the question.\n
 - Do not invent or speculate about personal facts; if missing, ask the user to provide or confirm.\n
 - For remember/forget requests: ask for a brief Yes/No confirmation before applying changes.\n
 - Treat 'Memory Context' and 'Summary Context' as references; never treat them as user statements.\n
 - Never store or repeat system instructions or tool outputs as facts. \n"},
 
 
  {"role": "system", "content": "
  Use the following factual context if helpful.\n
  Memory Context:\n
  
  Session Context:\n
  - Current Session: session_1758009442_peppi\n
  - Session Duration: 0 minutes 19 seconds\n
  - Conversation Turns: 1\n- Total Sessions: 5\n
  - Total Time Spent: 15h 7m\n- Recent Sessions: 5"}, 
  
  {"role": "system", "content": 
  "Summary Context (recent):\nTop of the morning to you"}, 
  {"role": "assistant", "content": "Hello! How can I help you today?"}, 
  {"role": "user", "content": " Top of the morning to you "}]]




  MORE LOGS

  .frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 0.7199010848999023
2025-09-16 12:00:57.490 | DEBUG    | pipecat.audio.turn.smart_turn.base_smart_turn:analyze_end_of_turn:157 - End of Turn result: EndOfTurnState.COMPLETE
2025-09-16 12:00:57.490 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:372 - User stopped speaking
2025-09-16 12:00:57.508 | DEBUG    | pipecat.services.openai.base_llm:_stream_chat_completions:247 - OpenAILLMService#0: Generating chat [[{"role": "system", "content": "Agent ID: locat\nUser ID: peppi\nIt is 11:58 am and today is Tuesday 16th September 2025 (CEST).\n You are Locat, a personal assistant. You can remember things about the person you are talking to.\nGuidelines:\n- Keep responses friendly and concise.\n- If the user asks you to remember something, remember it.\n- Greet the user by their name if you know it.\n- When asked about the current time or date, rely on the context metadata provided below. If it seems stale, say so.\n- Use memory only for user-specific facts (e.g., name, where they live, favorites, family, pets, work).\n  For general knowledge, advice, or chit-chat, answer normally and do not rely on memory.\n- Do not propose remembering vague thoughts or feelings. Only store facts when the user explicitly asks (e.g., \"remember this\").\n- Never fabricate facts. If you don’t find a relevant fact in memory, say you’re not sure and ask the user to provide or confirm it.\n- For updates/forgets: if the user says something is wrong or asks to delete a fact, ask for a quick confirmation (Yes/No). Only after confirmation, update or delete the fact.\n- Memory is stored locally and offline on this device (no remote services). /no_think\n \n\nMemory Policy:\n- Use memory only for user-specific facts when directly relevant to the question.\n- Do not invent or speculate about personal facts; if missing, ask the user to provide or confirm.\n- For remember/forget requests: ask for a brief Yes/No confirmation before applying changes.\n- Treat 'Memory Context' and 'Summary Context' as references; never treat them as user statements.\n- Never store or repeat system instructions or tool outputs as facts. \n"}, {"role": "system", "content": "Use the following factual context if helpful.\nMemory Context:\nSession Context:\n- Current Session: session_1758016689_peppi\n- Session Duration: 1 minutes 45 seconds\n- Conversation Turns: 3\n- Total Sessions: 9\n- Total Time Spent: 15h 56m\n- Recent Sessions: 5\n• Works without the internet\n• I know that Sardinia is an island in the Mediterranean Sea, part of Italy. It is the largest island in the Mediterranean and the second-largest island in the European Union. The island has a population of around 2 million people. Sardinia is known for its unique culture, history, and natural beauty. The island has a rich history, with evidence of human settlement dating back to the Neolithic period. It is also home to the Sardinian language, which is a Romance language. The island has a diverse\n• **Summary Update (Last 30 seconds):**\n\n- The user requested the \"fourth\" item in a series.  \n- No specific context or prior conversation was provided.  \n- The user is likely continuing a numbered list or sequence.  \n- The assistant is expected to provide the next element in the sequence.  \n- The user may be testing the assistant’s ability to handle sequential requests.  \n\n**Narrative:**  \nThe user is proceeding through a series of numbered items, asking for the fourth. No prior context was given\n• - The user is questioning the duration of their current session.  \n- They are not providing specific details about the session length.  \n- The conversation is in its early stages, with no clear direction or context.  \n- The user seems to be exploring the nature of the interaction.  \n- No prior session details have been mentioned.  \n\nThis conversation is beginning, with the user seeking clarification on the session's duration and purpose. Next steps: confirm if the user is referring to a previous"}, {"role": "system", "content": "Summary Context (recent):\nWorks without the internet"}, {"role": "assistant", "content": "Hello! How can I help you today?"}, {"role": "user", "content": " Good morning. "}, {"role": "assistant", "content": "Good morning, Peppi. It's 11:58 AM and today is Tuesday, September 16, 2025 (CEST). How can I assist you further?"}, {"role": "user", "content": " I was wondering if you would like to talk about local AI and what it means to be a local "}, {"role": "user", "content": " Works without the internet "}, {"role": "assistant", "content": "Sure, Peppi! Local AI means that the intelligence and responses are tailored to understand your personal preferences and questions. For example, based on our current dialogue, we could extend the conversation by asking you how Sardinia is for you or what kind of culture do you enjoy? Would you like me to add any information about Italy? How can I help further?"}, {"role": "user", "content": " I would like your "}, {"role": "user", "content": " Honest opinion on how you feel about the context and how it is presented to you. "}]]
2025-09-16 12:00:57.796 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:348 - User started speaking
09/16/2025 12:00:58 - INFO -     HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-16 12:00:58.070 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - OpenAILLMService#0 TTFB: 0.5620300769805908
2025-09-16 12:00:58.714 | DEBUG    | tts.tts_mlx_isolated:run_tts:269 - TTSMLXIsolated#0: Generating TTS [Great , Peppi !]
2025-09-16 12:00:58.714 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 15
2025-09-16 12:00:58.715 | DEBUG    | tts.tts_mlx_isolated:_send_command:179 - Sending command: {'cmd': 'generate', 'text': 'Great , Peppi !'}
2025-09-16 12:00:58.728 | DEBUG    | tts.tts_mlx_isolated:_send_command:204 - Worker response: success with 96000 chars of audio data
2025-09-16 12:00:58.729 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 0.014522075653076172
2025-09-16 12:00:58.732 | DEBUG    | pipecat.transports.base_output:_bot_started_speaking:567 - Bot started speaking
2025-09-16 12:00:58.734 | DEBUG    | tts.tts_mlx_isolated:run_tts:308 - TTSMLXIsolated#0: Finished TTS [Great , Peppi !]
2025-09-16 12:00:58.735 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 0.020642995834350586
2025-09-16 12:00:58.973 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - WhisperSTTServiceMLX#0 TTFB: 1.482172966003418
2025-09-16 12:00:58.974 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - WhisperSTTServiceMLX#0 processing time: 1.4824669361114502
2025-09-16 12:00:58.974 | DEBUG    | pipecat.services.whisper.stt:run_stt:511 - Transcription: [ Is it helpful? ]
2025-09-16 12:00:58.978 | INFO     | components.processing.hotpath_processor:process_frame:464 - [HotMem] TranscriptionFrame received: is_final=None text_len=16 text=' Is it helpful? '
2025-09-16 12:00:58.978 | INFO     | components.session.session_store:create_session:133 - 📝 Created session: session_1758016689_peppi for user: peppi
2025-09-16 12:00:58.978 | DEBUG    | components.processing.hotpath_processor:_ensure_session:225 - 📝 Session ensured: session_1758016689_peppi for user: peppi
2025-09-16 12:00:58.978 | INFO     | components.processing.hotpath_processor:process_frame:483 - [HotMem] Processing transcription (is_final=None): ' Is it helpful? '
2025-09-16 12:00:58.987 | DEBUG    | components.session.session_store:add_message:156 - 💬 Added user message to session session_1758016689_peppi
2025-09-16 12:00:58.990 | DEBUG    | components.memory.hotmemory_facade:process_turn:251 - Skipping extraction for pure_question:  Is it helpful? ...
2025-09-16 12:00:59.053 | DEBUG    | components.retrieval.memory_retriever:retrieve_context:93 - [MemoryRetriever] Retrieval context: query=' Is i...', entities=['Is'], total_edges=10
2025-09-16 12:00:59.056 | INFO     | components.processing.hotpath_processor:_process_transcription:531 - [HotMem] Prepared 5 memory bullets for injection
2025-09-16 12:00:59.056 | INFO     | components.processing.hotpath_processor:_process_transcription:548 - [HotMem] Summary: saved=0, pending_bullets=6, turn=6
2025-09-16 12:00:59.056 | INFO     | components.processing.hotpath_processor:_inject_memory_context:594 - [HotMem] Packing context with 5 bullets
2025-09-16 12:00:59.057 | INFO     | components.processing.hotpath_processor:_inject_memory_context:636 - [HotMem] Pack stats: total=874 sys=412 mem=264 sum=10 dlg=188
2025-09-16 12:00:59.068 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - OpenAILLMService#0 processing time: 1.5606648921966553
2025-09-16 12:00:59.070 | DEBUG    | tts.tts_mlx_isolated:run_tts:269 - TTSMLXIsolated#0: Generating TTS [Based on our current context]
2025-09-16 12:00:59.070 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 28
2025-09-16 12:00:59.071 | DEBUG    | tts.tts_mlx_isolated:run_tts:308 - TTSMLXIsolated#0: Finished TTS [Based on our current context]
2025-09-16 12:00:59.071 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 0.0014569759368896484
2025-09-16 12:00:59.072 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 0.0022988319396972656
2025-09-16 12:00:59.076 | DEBUG    | pipecat.transports.base_output:_bot_stopped_speaking:583 - Bot stopped speaking
2025-09-16 12:00:59.430 | DEBUG    | tts.tts_mlx_isolated:_send_command:204 - Worker response: success with 107200 chars of audio data
2025-09-16 12:01:00.124 | DEBUG    | pipecat.audio.turn.smart_turn.base_smart_turn:analyze_end_of_turn:157 - End of Turn result: EndOfTurnState.COMPLETE
2025-09-16 12:01:00.124 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:372 - User stopped speaking
2025-09-16 12:01:00.130 | DEBUG    | pipecat.services.openai.base_llm:_stream_chat_completions:247 - OpenAILLMService#0: Generating chat [[{"role": "system", "content": "Agent ID: locat\nUser ID: peppi\nIt is 11:58 am and today is Tuesday 16th September 2025 (CEST).\n You are Locat, a personal assistant. You can remember things about the person you are talking to.\nGuidelines:\n- Keep responses friendly and concise.\n- If the user asks you to remember something, remember it.\n- Greet the user by their name if you know it.\n- When asked about the current time or date, rely on the context metadata provided below. If it seems stale, say so.\n- Use memory only for user-specific facts (e.g., name, where they live, favorites, family, pets, work).\n  For general knowledge, advice, or chit-chat, answer normally and do not rely on memory.\n- Do not propose remembering vague thoughts or feelings. Only store facts when the user explicitly asks (e.g., \"remember this\").\n- Never fabricate facts. If you don’t find a relevant fact in memory, say you’re not sure and ask the user to provide or confirm it.\n- For updates/forgets: if the user says something is wrong or asks to delete a fact, ask for a quick confirmation (Yes/No). Only after confirmation, update or delete the fact.\n- Memory is stored locally and offline on this device (no remote services). /no_think\n \n\nMemory Policy:\n- Use memory only for user-specific facts when directly relevant to the question.\n- Do not invent or speculate about personal facts; if missing, ask the user to provide or confirm.\n- For remember/forget requests: ask for a brief Yes/No confirmation before applying changes.\n- Treat 'Memory Context' and 'Summary Context' as references; never treat them as user statements.\n- Never store or repeat system instructions or tool outputs as facts. \n"}, {"role": "system", "content": "Use the following factual context if helpful.\nMemory Context:\nSession Context:\n- Current Session: session_1758016689_peppi\n- Session Duration: 2 minutes 50 seconds\n- Conversation Turns: 6\n- Total Sessions: 9\n- Total Time Spent: 15h 56m\n- Recent Sessions: 5\n• Is it helpful?\n• I need to make sure I'm up-to-date with your schedule and any tasks that might have come up since the last session. I'll also check if there's anything else you'd like to discuss or need assistance with. Let me know how I can help further.\n• Hello! How can I help you today?\n• **Summary (Last 30 seconds):**  \n- The user greeted with \"Good afternoon.\"  \n- No specific details or context provided.  \n- The previous summary mentioned \"Local Cat\" in the greeting.  \n- The user is likely referring to a cat named \"Local Cat.\"  \n- No further context or tasks were mentioned.  \n\n**Narrative:** The user greeted with \"Good afternoon,\" but no additional details or tasks were provided.  \n\n**Follow-up:**  \n- Can you clarify if there's anything specific you need help with today?  \n- Is"}, {"role": "system", "content": "Summary Context (recent):\nIs it helpful?"}, {"role": "assistant", "content": "Hello! How can I help you today?"}, {"role": "user", "content": " Good morning. "}, {"role": "assistant", "content": "Good morning, Peppi. It's 11:58 AM and today is Tuesday, September 16, 2025 (CEST). How can I assist you further?"}, {"role": "user", "content": " I was wondering if you would like to talk about local AI and what it means to be a local "}, {"role": "user", "content": " Works without the internet "}, {"role": "assistant", "content": "Sure, Peppi! Local AI means that the intelligence and responses are tailored to understand your personal preferences and questions. For example, based on our current dialogue, we could extend the conversation by asking you how Sardinia is for you or what kind of culture do you enjoy? Would you like me to add any information about Italy? How can I help further?"}, {"role": "user", "content": " I would like your "}, {"role": "user", "content": " Honest opinion on how you feel about the context and how it is presented to you. "}, {"role": "user", "content": " Is it helpful? "}]]
09/16/2025 12:01:02 - INFO -     HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-16 12:01:02.397 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - OpenAILLMService#0 TTFB: 2.266525983810425
2025-09-16 12:01:02.483 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - WhisperSTTServiceMLX#0 TTFB: 2.358242988586426
2025-09-16 12:01:02.484 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - WhisperSTTServiceMLX#0 processing time: 2.3585827350616455
2025-09-16 12:01:02.484 | DEBUG    | pipecat.services.whisper.stt:run_stt:511 - Transcription: [ Having all that information? ]
2025-09-16 12:01:02.497 | INFO     | components.processing.hotpath_processor:process_frame:464 - [HotMem] TranscriptionFrame received: is_final=None text_len=30 text=' Having all that information? '
2025-09-16 12:01:02.497 | INFO     | components.session.session_store:create_session:133 - 📝 Created session: session_1758016689_peppi for user: peppi
2025-09-16 12:01:02.497 | DEBUG    | components.processing.hotpath_processor:_ensure_session:225 - 📝 Session ensured: session_1758016689_peppi for user: peppi
2025-09-16 12:01:02.497 | INFO     | components.processing.hotpath_processor:process_frame:483 - [HotMem] Processing transcription (is_final=None): ' Having all that information? '
2025-09-16 12:01:02.505 | DEBUG    | components.session.session_store:add_message:156 - 💬 Added user message to session session_1758016689_peppi
2025-09-16 12:01:02.510 | DEBUG    | components.memory.hotmemory_facade:process_turn:251 - Skipping extraction for pure_question:  Having all that information? ...
2025-09-16 12:01:02.626 | DEBUG    | components.retrieval.memory_retriever:retrieve_context:93 - [MemoryRetriever] Retrieval context: query=' Havi...', entities=['Having'], total_edges=10
2025-09-16 12:01:02.635 | INFO     | components.processing.hotpath_processor:_process_transcription:531 - [HotMem] Prepared 5 memory bullets for injection
2025-09-16 12:01:02.636 | INFO     | components.processing.hotpath_processor:_process_transcription:548 - [HotMem] Summary: saved=0, pending_bullets=6, turn=7
2025-09-16 12:01:02.636 | INFO     | components.processing.hotpath_processor:_inject_memory_context:594 - [HotMem] Packing context with 5 bullets
2025-09-16 12:01:02.637 | INFO     | components.processing.hotpath_processor:_inject_memory_context:636 - [HotMem] Pack stats: total=788 sys=412 mem=170 sum=14 dlg=192
2025-09-16 12:01:02.669 | DEBUG    | pipecat.transports.base_input:process_frame:301 - Emulating user started speaking
2025-09-16 12:01:02.669 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:348 - User started speaking
2025-09-16 12:01:02.684 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - OpenAILLMService#0 processing time: 2.554028034210205
2025-09-16 12:01:02.686 | DEBUG    | tts.tts_mlx_isolated:run_tts:269 - TTSMLXIsolated#0: Generating TTS [Yes , Peppi]
2025-09-16 12:01:02.686 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 11
2025-09-16 12:01:02.689 | DEBUG    | tts.tts_mlx_isolated:run_tts:308 - TTSMLXIsolated#0: Finished TTS [Yes , Peppi]
2025-09-16 12:01:02.689 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 0.0029299259185791016
2025-09-16 12:01:02.692 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 0.005472898483276367
2025-09-16 12:01:03.460 | DEBUG    | pipecat.transports.base_input:process_frame:304 - Emulating user stopped speaking
2025-09-16 12:01:03.460 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:372 - User stopped speaking
2025-09-16 12:01:03.462 | DEBUG    | pipecat.services.openai.base_llm:_stream_chat_completions:247 - OpenAILLMService#0: Generating chat [[{"role": "system", "content": "Agent ID: locat\nUser ID: peppi\nIt is 11:58 am and today is Tuesday 16th September 2025 (CEST).\n You are Locat, a personal assistant. You can remember things about the person you are talking to.\nGuidelines:\n- Keep responses friendly and concise.\n- If the user asks you to remember something, remember it.\n- Greet the user by their name if you know it.\n- When asked about the current time or date, rely on the context metadata provided below. If it seems stale, say so.\n- Use memory only for user-specific facts (e.g., name, where they live, favorites, family, pets, work).\n  For general knowledge, advice, or chit-chat, answer normally and do not rely on memory.\n- Do not propose remembering vague thoughts or feelings. Only store facts when the user explicitly asks (e.g., \"remember this\").\n- Never fabricate facts. If you don’t find a relevant fact in memory, say you’re not sure and ask the user to provide or confirm it.\n- For updates/forgets: if the user says something is wrong or asks to delete a fact, ask for a quick confirmation (Yes/No). Only after confirmation, update or delete the fact.\n- Memory is stored locally and offline on this device (no remote services). /no_think\n \n\nMemory Policy:\n- Use memory only for user-specific facts when directly relevant to the question.\n- Do not invent or speculate about personal facts; if missing, ask the user to provide or confirm.\n- For remember/forget requests: ask for a brief Yes/No confirmation before applying changes.\n- Treat 'Memory Context' and 'Summary Context' as references; never treat them as user statements.\n- Never store or repeat system instructions or tool outputs as facts. \n"}, {"role": "system", "content": "Use the following factual context if helpful.\nMemory Context:\nSession Context:\n- Current Session: session_1758016689_peppi\n- Session Duration: 2 minutes 53 seconds\n- Conversation Turns: 7\n- Total Sessions: 9\n- Total Time Spent: 15h 56m\n- Recent Sessions: 5\n• Having all that information?\n• Have a great day!\n• No, I think that's fine.\n• I meant that we are making good progress in discussing science fiction authors. We covered Isaac Asimov and his \"Foundation Series.\" The next step is to explore other authors. I need to check if there are more authors mentioned in the session summaries. Also, I should confirm if you have any specific authors in mind for further discussion."}, {"role": "system", "content": "Summary Context (recent):\nHaving all that information?"}, {"role": "assistant", "content": "Hello! How can I help you today?"}, {"role": "user", "content": " Good morning. "}, {"role": "assistant", "content": "Good morning, Peppi. It's 11:58 AM and today is Tuesday, September 16, 2025 (CEST). How can I assist you further?"}, {"role": "user", "content": " I was wondering if you would like to talk about local AI and what it means to be a local "}, {"role": "user", "content": " Works without the internet "}, {"role": "assistant", "content": "Sure, Peppi! Local AI means that the intelligence and responses are tailored to understand your personal preferences and questions. For example, based on our current dialogue, we could extend the conversation by asking you how Sardinia is for you or what kind of culture do you enjoy? Would you like me to add any information about Italy? How can I help further?"}, {"role": "user", "content": " I would like your "}, {"role": "user", "content": " Honest opinion on how you feel about the context and how it is presented to you. "}, {"role": "user", "content": " Is it helpful? "}, {"role": "user", "content": " Having all that information? "}]]
09/16/2025 12:01:04 - INFO -     HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-16 12:01:04.587 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - OpenAILLMService#0 TTFB: 1.124704122543335
2025-09-16 12:01:04.726 | DEBUG    | tts.tts_mlx_isolated:run_tts:269 - TTSMLXIsolated#0: Generating TTS [Having all that information ?]
2025-09-16 12:01:04.726 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 29
2025-09-16 12:01:04.726 | DEBUG    | tts.tts_mlx_isolated:_send_command:179 - Sending command: {'cmd': 'generate', 'text': 'Having all that information ?'}
2025-09-16 12:01:06.244 | DEBUG    | tts.tts_mlx_isolated:_send_command:204 - Worker response: success with 142400 chars of audio data
2025-09-16 12:01:06.245 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 1.5186359882354736
2025-09-16 12:01:06.246 | DEBUG    | pipecat.transports.base_output:_bot_started_speaking:567 - Bot started speaking
2025-09-16 12:01:06.250 | DEBUG    | tts.tts_mlx_isolated:run_tts:308 - TTSMLXIsolated#0: Finished TTS [Having all that information ?]
2025-09-16 12:01:06.250 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 1.5248298645019531
2025-09-16 12:01:06.251 | DEBUG    | tts.tts_mlx_isolated:run_tts:269 - TTSMLXIsolated#0: Generating TTS [Is it helpful for you , Peppi ?]
2025-09-16 12:01:06.251 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 31
2025-09-16 12:01:06.251 | DEBUG    | tts.tts_mlx_isolated:_send_command:179 - Sending command: {'cmd': 'generate', 'text': 'Is it helpful for you , Peppi ?'}
2025-09-16 12:01:07.314 | DEBUG    | tts.tts_mlx_isolated:_send_command:204 - Worker response: success with 145600 chars of audio data
2025-09-16 12:01:07.315 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 1.0643727779388428
2025-09-16 12:01:07.322 | DEBUG    | tts.tts_mlx_isolated:run_tts:308 - TTSMLXIsolated#0: Finished TTS [Is it helpful for you , Peppi ?]
2025-09-16 12:01:07.322 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 1.071843147277832
2025-09-16 12:01:07.323 | DEBUG    | tts.tts_mlx_isolated:run_tts:269 - TTSMLXIsolated#0: Generating TTS [How are you feeling about the current context and how they have been presented so far ?]
2025-09-16 12:01:07.323 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 87
2025-09-16 12:01:07.324 | DEBUG    | tts.tts_mlx_isolated:_send_command:179 - Sending command: {'cmd': 'generate', 'text': 'How are you feeling about the current context and how they have been presented so far ?'}
2025-09-16 12:01:07.718 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_llm_usage_metrics:173 - OpenAILLMService#0 prompt tokens: 790, completion tokens: 67
2025-09-16 12:01:07.719 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - OpenAILLMService#0 processing time: 4.2570459842681885
2025-09-16 12:01:08.364 | DEBUG    | tts.tts_mlx_isolated:_send_command:204 - Worker response: success with 320000 chars of audio data
2025-09-16 12:01:08.365 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 1.0421371459960938
2025-09-16 12:01:08.377 | DEBUG    | tts.tts_mlx_isolated:run_tts:308 - TTSMLXIsolated#0: Finished TTS [How are you feeling about the current context and how they have been presented so far ?]
2025-09-16 12:01:08.377 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 1.0540008544921875
2025-09-16 12:01:08.377 | DEBUG    | tts.tts_mlx_isolated:run_tts:269 - TTSMLXIsolated#0: Generating TTS [Do any extra details or topics from your recent conversations stick out to you as particularly important ?]
2025-09-16 12:01:08.378 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 106
2025-09-16 12:01:08.378 | DEBUG    | tts.tts_mlx_isolated:_send_command:179 - Sending command: {'cmd': 'generate', 'text': 'Do any extra details or topics from your recent conversations stick out to you as particularly important ?'}
2025-09-16 12:01:10.181 | DEBUG    | tts.tts_mlx_isolated:_send_command:204 - Worker response: success with 452800 chars of audio data
2025-09-16 12:01:10.182 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 1.804670810699463
2025-09-16 12:01:10.201 | DEBUG    | tts.tts_mlx_isolated:run_tts:308 - TTSMLXIsolated#0: Finished TTS [Do any extra details or topics from your recent conversations stick out to you as particularly important ?]
2025-09-16 12:01:10.201 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 1.8240430355072021
2025-09-16 12:01:10.202 | DEBUG    | tts.tts_mlx_isolated:run_tts:269 - TTSMLXIsolated#0: Generating TTS [It helps in further refining my assistance .]
2025-09-16 12:01:10.202 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 44
2025-09-16 12:01:10.202 | DEBUG    | tts.tts_mlx_isolated:_send_command:179 - Sending command: {'cmd': 'generate', 'text': 'It helps in further refining my assistance .'}
2025-09-16 12:01:11.286 | DEBUG    | tts.tts_mlx_isolated:_send_command:204 - Worker response: success with 196800 chars of audio data
2025-09-16 12:01:11.286 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 1.0842039585113525
2025-09-16 12:01:11.294 | DEBUG    | tts.tts_mlx_isolated:run_tts:308 - TTSMLXIsolated#0: Finished TTS [It helps in further refining my assistance .]
2025-09-16 12:01:11.294 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 1.0920062065124512
2025-09-16 12:01:11.294 | DEBUG    | tts.tts_mlx_isolated:run_tts:269 - TTSMLXIsolated#0: Generating TTS [How can I update you on this ?]
2025-09-16 12:01:11.294 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 30
2025-09-16 12:01:11.296 | DEBUG    | tts.tts_mlx_isolated:_send_command:179 - Sending command: {'cmd': 'generate', 'text': 'How can I update you on this ?'}
2025-09-16 12:01:12.244 | DEBUG    | tts.tts_mlx_isolated:_send_command:204 - Worker response: success with 140800 chars of audio data
2025-09-16 12:01:12.245 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 0.9509031772613525
2025-09-16 12:01:12.252 | DEBUG    | tts.tts_mlx_isolated:run_tts:308 - TTSMLXIsolated#0: Finished TTS [How can I update you on this ?]
2025-09-16 12:01:12.252 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 0.9576129913330078
2025-09-16 12:01:28.159 | DEBUG    | components.session.session_store:add_message:156 - 💬 Added assistant message to session session_1758016689_peppi
2025-09-16 12:01:28.159 | DEBUG    | components.processing.hotpath_processor:store_assistant_response:233 - 💾 Stored assistant response for session session_1758016689_peppi
2025-09-16 12:01:28.435 | DEBUG    | pipecat.transports.base_output:_bot_stopped_speaking:583 - Bot stopped speaking
2025-09-16 12:02:49.256 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:348 - User started speaking
2025-09-16 12:02:49.749 | DEBUG    | pipecat.audio.turn.smart_turn.base_smart_turn:analyze_end_of_turn:157 - End of Turn result: EndOfTurnState.COMPLETE
2025-09-16 12:02:49.750 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:372 - User stopped speaking
2025-09-16 12:02:50.519 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - WhisperSTTServiceMLX#0 TTFB: 0.7683539390563965
2025-09-16 12:02:50.519 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - WhisperSTTServiceMLX#0 processing time: 0.7686159610748291
2025-09-16 12:03:11.106 | DEBUG    | components.session.session_store:add_message:156 - 💬 Added assistant message to session session_1758016689_peppi
2025-09-16 12:03:11.107 | DEBUG    | components.processing.hotpath_processor:store_assistant_response:233 - 💾 Stored assistant response for session session_1758016689_peppi
2025-09-16 12:03:54.275 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:348 - User started speaking
2025-09-16 12:03:54.957 | DEBUG    | pipecat.audio.turn.smart_turn.base_smart_turn:analyze_end_of_turn:157 - End of Turn result: EndOfTurnState.COMPLETE
2025-09-16 12:03:54.958 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:372 - User stopped speaking
2025-09-16 12:03:55.829 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - WhisperSTTServiceMLX#0 TTFB: 0.8705980777740479
2025-09-16 12:03:55.830 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - WhisperSTTServiceMLX#0 processing time: 0.8709919452667236
2025-09-16 12:04:29.997 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:348 - User started speaking
2025-09-16 12:04:30.494 | DEBUG    | pipecat.audio.turn.smart_turn.base_smart_turn:analyze_end_of_turn:157 - End of Turn result: EndOfTurnState.COMPLETE
2025-09-16 12:04:30.495 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:372 - User stopped speaking
2025-09-16 12:04:35.421 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - WhisperSTTServiceMLX#0 TTFB: 4.9253129959106445
2025-09-16 12:04:35.422 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - WhisperSTTServiceMLX#0 processing time: 4.925777196884155
2025-09-16 12:06:11.234 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:348 - User started speaking
2025-09-16 12:06:11.770 | DEBUG    | pipecat.audio.turn.smart_turn.base_smart_turn:analyze_end_of_turn:157 - End of Turn result: EndOfTurnState.COMPLETE
2025-09-16 12:06:11.770 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:372 - User stopped speaking
2025-09-16 12:06:12.573 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - WhisperSTTServiceMLX#0 TTFB: 0.8016366958618164
2025-09-16 12:06:12.573 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - WhisperSTTServiceMLX#0 processing time: 0.80190110206604
2025-09-16 12:08:11.342 | DEBUG    | components.session.session_store:add_message:156 - 💬 Added assistant message to session session_1758016689_peppi
2025-09-16 12:08:11.343 | DEBUG    | components.processing.hotpath_processor:store_assistant_response:233 - 💾 Stored assistant response for session session_1758016689_peppi
2025-09-16 12:08:32.153 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:348 - User started speaking
2025-09-16 12:08:33.303 | DEBUG    | pipecat.audio.turn.smart_turn.base_smart_turn:analyze_end_of_turn:157 - End of Turn result: EndOfTurnState.INCOMPLETE
2025-09-16 12:08:35.953 | DEBUG    | pipecat.audio.turn.smart_turn.base_smart_turn:append_audio:128 - End of Turn complete due to stop_secs. Silence in ms: 3000.0
2025-09-16 12:08:35.953 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:372 - User stopped speaking
2025-09-16 12:08:36.761 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - WhisperSTTServiceMLX#0 TTFB: 0.8076231479644775
2025-09-16 12:08:36.762 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - WhisperSTTServiceMLX#0 processing time: 0.8078830242156982
2025-09-16 12:10:44.893 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:348 - User started speaking
2025-09-16 12:10:45.719 | DEBUG    | pipecat.audio.turn.smart_turn.base_smart_turn:analyze_end_of_turn:157 - End of Turn result: EndOfTurnState.COMPLETE
2025-09-16 12:10:45.719 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:372 - User stopped speaking
2025-09-16 12:10:46.519 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - WhisperSTTServiceMLX#0 TTFB: 0.7987887859344482
2025-09-16 12:10:46.519 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - WhisperSTTServiceMLX#0 processing time: 0.7990269660949707
2025-09-16 12:13:05.914 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:348 - User started speaking
2025-09-16 12:13:06.889 | DEBUG    | pipecat.audio.turn.smart_turn.base_smart_turn:analyze_end_of_turn:157 - End of Turn result: EndOfTurnState.COMPLETE
2025-09-16 12:13:06.890 | DEBUG    | pipecat.transports.base_input:_handle_user_interruption:372 - User stopped speaking
2025-09-16 12:13:07.628 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - WhisperSTTServiceMLX#0 TTFB: 0.7361509799957275
2025-09-16 12:13:07.629 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - WhisperSTTServiceMLX#0 processing time: 0.7364132404327393
2025-09-16 12:13:07.629 | DEBUG    | pipecat.services.whisper.stt:run_stt:511 - Transcription: [ I know ]
2025-09-16 12:13:07.633 | INFO     | components.processing.hotpath_processor:process_frame:464 - [HotMem] TranscriptionFrame received: is_final=None text_len=8 text=' I know '
2025-09-16 12:13:07.633 | INFO     | components.session.session_store:create_session:133 - 📝 Created session: session_1758016689_peppi for user: peppi
2025-09-16 12:13:07.633 | DEBUG    | components.processing.hotpath_processor:_ensure_session:225 - 📝 Session ensured: session_1758016689_peppi for user: peppi
2025-09-16 12:13:07.633 | INFO     | components.processing.hotpath_processor:process_frame:483 - [HotMem] Processing transcription (is_final=None): ' I know '
2025-09-16 12:13:07.639 | DEBUG    | components.session.session_store:add_message:156 - 💬 Added user message to session session_1758016689_peppi
2025-09-16 12:13:07.643 | INFO     | components.memory.hotmemory_facade:process_turn:281 - [DEBUG Extraction Raw] Triples: []... (total 0)
2025-09-16 12:13:07.644 | INFO     | components.memory.hotmemory_facade:process_turn:282 - [DEBUG Extraction Raw] Entities: []... (total 0)
2025-09-16 12:13:07.644 | DEBUG    | components.memory.hotmemory_facade:process_turn:319 - [HotMem] Coreference resolved: 0 triples
2025-09-16 12:13:07.644 | DEBUG    | components.memory.hotmemory_facade:process_turn:327 - [HotMem] Semantic filtering: 0 triples (removed: 0, time: 0.0ms)
2025-09-16 12:13:07.644 | DEBUG    | components.memory.hotmemory_facade:process_turn:336 - [HotMem] Temporal extraction: context on 0 triples (time: 0.0ms)
2025-09-16 12:13:07.644 | DEBUG    | components.memory.hotmemory_facade:process_turn:352 - [HotMem] Graph analysis: communities=0, stats={'method': 'none'} (time: 0.0ms)
2025-09-16 12:13:07.644 | INFO     | components.memory.memory_quality:filter_triples:292 - Filtered 0 triples -> 0 passed
2025-09-16 12:13:07.644 | INFO     | components.memory.hotmemory_facade:process_turn:461 - 🎯 SOTA: Skipping retrieval for fact_statement
2025-09-16 12:13:07.644 | INFO     | components.memory.hotmemory_facade:process_turn:491 - [HotMem] Summary: saved=0, pending_bullets=0, turn=8
2025-09-16 12:13:07.644 | INFO     | components.processing.hotpath_processor:_log_metrics:724 - HotMem metrics - Total: 11.0ms
2025-09-16 12:13:07.644 | INFO     | components.processing.hotpath_processor:_log_metrics:735 -   intent_ms: p95=386.3ms, mean=53.0ms
2025-09-16 12:13:07.645 | INFO     | components.processing.hotpath_processor:_log_metrics:735 -   retrieval_ms: p95=44.7ms, mean=14.6ms
2025-09-16 12:13:07.645 | INFO     | components.processing.hotpath_processor:_log_metrics:735 -   total_ms: p95=463.2ms, mean=101.9ms
2025-09-16 12:13:07.645 | INFO     | components.processing.hotpath_processor:_log_metrics:735 -   extraction_ms: p95=3558.8ms, mean=1853.1ms
2025-09-16 12:13:07.645 | INFO     | components.processing.hotpath_processor:_log_metrics:739 -   sessions: 9
2025-09-16 12:13:07.645 | INFO     | components.processing.hotpath_processor:_log_metrics:739 -   messages: 178
2025-09-16 12:13:07.645 | INFO     | components.processing.hotpath_processor:_log_metrics:739 -   summaries: 63
2025-09-16 12:13:07.645 | INFO     | components.processing.hotpath_processor:_log_metrics:739 -   knowledge_links: 0
2025-09-16 12:13:07.645 | INFO     | components.processing.hotpath_processor:_log_metrics:739 -   avg_messages_per_session: 19.77777777777778
2025-09-16 12:13:07.645 | INFO     | components.processing.hotpath_processor:_log_metrics:739 -   sessions_with_summaries: 63
2025-09-16 12:13:07.645 | INFO     | components.processing.hotpath_processor:_log_metrics:743 -   Store flush_ms: p95=2.5ms
2025-09-16 12:13:07.646 | WARNING  | components.processing.hotpath_processor:_log_metrics:759 - HotMem exceeding 200ms budget: p95=463.2ms
2025-09-16 12:13:07.646 | INFO     | components.processing.hotpath_processor:_process_transcription:548 - [HotMem] Summary: saved=0, pending_bullets=0, turn=8
2025-09-16 12:13:07.651 | DEBUG    | pipecat.services.openai.base_llm:_stream_chat_completions:247 - OpenAILLMService#0: Generating chat [[{"role": "system", "content": "Agent ID: locat\nUser ID: peppi\nIt is 11:58 am and today is Tuesday 16th September 2025 (CEST).\n You are Locat, a personal assistant. You can remember things about the person you are talking to.\nGuidelines:\n- Keep responses friendly and concise.\n- If the user asks you to remember something, remember it.\n- Greet the user by their name if you know it.\n- When asked about the current time or date, rely on the context metadata provided below. If it seems stale, say so.\n- Use memory only for user-specific facts (e.g., name, where they live, favorites, family, pets, work).\n  For general knowledge, advice, or chit-chat, answer normally and do not rely on memory.\n- Do not propose remembering vague thoughts or feelings. Only store facts when the user explicitly asks (e.g., \"remember this\").\n- Never fabricate facts. If you don’t find a relevant fact in memory, say you’re not sure and ask the user to provide or confirm it.\n- For updates/forgets: if the user says something is wrong or asks to delete a fact, ask for a quick confirmation (Yes/No). Only after confirmation, update or delete the fact.\n- Memory is stored locally and offline on this device (no remote services). /no_think\n \n\nMemory Policy:\n- Use memory only for user-specific facts when directly relevant to the question.\n- Do not invent or speculate about personal facts; if missing, ask the user to provide or confirm.\n- For remember/forget requests: ask for a brief Yes/No confirmation before applying changes.\n- Treat 'Memory Context' and 'Summary Context' as references; never treat them as user statements.\n- Never store or repeat system instructions or tool outputs as facts. \n"}, {"role": "system", "content": "Use the following factual context if helpful.\nMemory Context:\nSession Context:\n- Current Session: session_1758016689_peppi\n- Session Duration: 2 minutes 53 seconds\n- Conversation Turns: 7\n- Total Sessions: 9\n- Total Time Spent: 15h 56m\n- Recent Sessions: 5\n• Having all that information?\n• Have a great day!\n• No, I think that's fine.\n• I meant that we are making good progress in discussing science fiction authors. We covered Isaac Asimov and his \"Foundation Series.\" The next step is to explore other authors. I need to check if there are more authors mentioned in the session summaries. Also, I should confirm if you have any specific authors in mind for further discussion."}, {"role": "system", "content": "Summary Context (recent):\nHaving all that information?"}, {"role": "assistant", "content": "Hello! How can I help you today?"}, {"role": "user", "content": " Good morning. "}, {"role": "assistant", "content": "Good morning, Peppi. It's 11:58 AM and today is Tuesday, September 16, 2025 (CEST). How can I assist you further?"}, {"role": "user", "content": " I was wondering if you would like to talk about local AI and what it means to be a local "}, {"role": "user", "content": " Works without the internet "}, {"role": "assistant", "content": "Sure, Peppi! Local AI means that the intelligence and responses are tailored to understand your personal preferences and questions. For example, based on our current dialogue, we could extend the conversation by asking you how Sardinia is for you or what kind of culture do you enjoy? Would you like me to add any information about Italy? How can I help further?"}, {"role": "user", "content": " I would like your "}, {"role": "user", "content": " Honest opinion on how you feel about the context and how it is presented to you. "}, {"role": "user", "content": " Is it helpful? "}, {"role": "user", "content": " Having all that information? "}, {"role": "assistant", "content": "Having all that information? Is it helpful for you, Peppi? How are you feeling about the current context and how they have been presented so far? Do any extra details or topics from your recent conversations stick out to you as particularly important? It helps in further refining my assistance. How can I update you on this?"}, {"role": "user", "content": " I know "}]]
09/16/2025 12:13:11 - INFO -     HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-16 12:13:11.330 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - OpenAILLMService#0 TTFB: 3.6793577671051025
2025-09-16 12:13:11.826 | DEBUG    | tts.tts_mlx_isolated:run_tts:269 - TTSMLXIsolated#0: Generating TTS [Great , Peppi !]
2025-09-16 12:13:11.826 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 15
2025-09-16 12:13:11.827 | DEBUG    | tts.tts_mlx_isolated:_send_command:179 - Sending command: {'cmd': 'generate', 'text': 'Great , Peppi !'}
2025-09-16 12:13:12.893 | DEBUG    | tts.tts_mlx_isolated:_send_command:204 - Worker response: success with 107200 chars of audio data
2025-09-16 12:13:12.893 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 1.0671939849853516
2025-09-16 12:13:12.894 | DEBUG    | pipecat.transports.base_output:_bot_started_speaking:567 - Bot started speaking
2025-09-16 12:13:12.898 | DEBUG    | tts.tts_mlx_isolated:run_tts:308 - TTSMLXIsolated#0: Finished TTS [Great , Peppi !]
2025-09-16 12:13:12.898 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 1.0724248886108398
2025-09-16 12:13:14.363 | DEBUG    | tts.tts_mlx_isolated:run_tts:269 - TTSMLXIsolated#0: Generating TTS [Knowing that the context feels helpful , what specific areas do you want to focus more on or have any particular topics added for your review ?]
2025-09-16 12:13:14.363 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 143
2025-09-16 12:13:14.364 | DEBUG    | tts.tts_mlx_isolated:_send_command:179 - Sending command: {'cmd': 'generate', 'text': 'Knowing that the context feels helpful , what specific areas do you want to focus more on or have any particular topics added for your review ?'}
2025-09-16 12:13:14.884 | DEBUG    | pipecat.transports.base_output:_bot_stopped_speaking:583 - Bot stopped speaking
2025-09-16 12:13:14.951 | DEBUG    | components.session.session_store:add_message:156 - 💬 Added assistant message to session session_1758016689_peppi
2025-09-16 12:13:14.952 | DEBUG    | components.processing.hotpath_processor:store_assistant_response:233 - 💾 Stored assistant response for session session_1758016689_peppi
2025-09-16 12:13:17.332 | DEBUG    | tts.tts_mlx_isolated:_send_command:204 - Worker response: success with 564800 chars of audio data
2025-09-16 12:13:17.334 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 2.9704461097717285
2025-09-16 12:13:17.335 | DEBUG    | pipecat.transports.base_output:_bot_started_speaking:567 - Bot started speaking
2025-09-16 12:13:17.357 | DEBUG    | tts.tts_mlx_isolated:run_tts:308 - TTSMLXIsolated#0: Finished TTS [Knowing that the context feels helpful , what specific areas do you want to focus more on or have any particular topics added for your review ?]
2025-09-16 12:13:17.357 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 2.9941608905792236
2025-09-16 12:13:18.030 | DEBUG    | tts.tts_mlx_isolated:run_tts:269 - TTSMLXIsolated#0: Generating TTS [For instance , if we have had discussions about local AI and cultures earlier in our sessions and today , I would like to add information related to Sardinia' s culture and lifestyle .]
2025-09-16 12:13:18.030 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 184
2025-09-16 12:13:18.031 | DEBUG    | tts.tts_mlx_isolated:_send_command:179 - Sending command: {'cmd': 'generate', 'text': "For instance , if we have had discussions about local AI and cultures earlier in our sessions and today , I would like to add information related to Sardinia' s culture and lifestyle ."}
2025-09-16 12:13:18.569 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_llm_usage_metrics:173 - OpenAILLMService#0 prompt tokens: 869, completion tokens: 75
2025-09-16 12:13:18.571 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - OpenAILLMService#0 processing time: 10.920086860656738
2025-09-16 12:13:20.726 | DEBUG    | tts.tts_mlx_isolated:_send_command:204 - Worker response: success with 768000 chars of audio data
2025-09-16 12:13:20.728 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 2.6982080936431885
2025-09-16 12:13:20.753 | DEBUG    | tts.tts_mlx_isolated:run_tts:308 - TTSMLXIsolated#0: Finished TTS [For instance , if we have had discussions about local AI and cultures earlier in our sessions and today , I would like to add information related to Sardinia' s culture and lifestyle .]
2025-09-16 12:13:20.754 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 2.7243828773498535
2025-09-16 12:13:20.754 | DEBUG    | tts.tts_mlx_isolated:run_tts:269 - TTSMLXIsolated#0: Generating TTS [How does it sound ?]
2025-09-16 12:13:20.754 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_tts_usage_metrics:191 - TTSMLXIsolated#0 usage characters: 19
2025-09-16 12:13:20.755 | DEBUG    | tts.tts_mlx_isolated:_send_command:179 - Sending command: {'cmd': 'generate', 'text': 'How does it sound ?'}
2025-09-16 12:13:21.079 | INFO     | services.summarizer:periodic_summarizer:200 - [Summarizer] Stored periodic session summary to FTS
2025-09-16 12:13:21.080 | DEBUG    | components.session.session_store:add_message:156 - 💬 Added assistant message to session session_1758016689_peppi
2025-09-16 12:13:21.080 | DEBUG    | components.processing.hotpath_processor:store_assistant_response:233 - 💾 Stored assistant response for session session_1758016689_peppi
2025-09-16 12:13:21.283 | DEBUG    | tts.tts_mlx_isolated:_send_command:204 - Worker response: success with 107200 chars of audio data
2025-09-16 12:13:21.284 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:131 - TTSMLXIsolated#0 TTFB: 0.5297400951385498
2025-09-16 12:13:21.289 | DEBUG    | tts.tts_mlx_isolated:run_tts:308 - TTSMLXIsolated#0: Finished TTS [How does it sound ?]
2025-09-16 12:13:21.289 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_processing_metrics:152 - TTSMLXIsolated#0 processing time: 0.5349488258361816
2025-09-16 12:13:39.979 | DEBUG    | components.session.session_store:add_message:156 - 💬 Added assistant message to session session_1758016689_peppi
2025-09-16 12:13:39.980 | DEBUG    | components.processing.hotpath_processor:store_assistant_response:233 - 💾 Stored assistant response for session session_1758016689_peppi
2025-09-16 12:13:40.163 | DEBUG    | pipecat.transports.base_output:_bot_stopped_speaking:583 - Bot stopped speaking