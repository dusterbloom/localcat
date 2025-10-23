// Siri TTS Streaming Sidecar for LocalCat
// Supports both file-based WAV output and PCM streaming for Pipecat integration
import AVFoundation
import Accelerate
import Darwin

// MARK: - Command Line Arguments

struct Args {
    var mode: Mode = .file
    var text: String = ""
    var outputPath: String = ""
    var language: String? = nil
    var voiceId: String? = nil
    var useSystemVoice: Bool = false
    var rate: Float = AVSpeechUtteranceDefaultSpeechRate
    var pitch: Float = 1.0
    var targetRate: Int = 24000  // Target sample rate (16000 for WebRTC, 24000 native)

    enum Mode {
        case file      // Write WAV to file
        case stream    // Stream raw PCM to stdout
    }
}

func parseArgs() -> Args? {
    var args = Args()
    var it = CommandLine.arguments.dropFirst().makeIterator()

    while let arg = it.next() {
        switch arg {
        case "--print-default-voice-id":
            if let vid = systemDefaultVoiceIdentifier() ?? voiceOverDefaultVoiceIdentifier() {
                print(vid)
                exit(0)
            } else {
                exit(1)
            }
        case "--stream-pcm":
            args.mode = .stream
        case "--text":
            args.text = it.next() ?? ""
        case "--lang", "--language":
            args.language = it.next()
        case "--voice-id":
            args.voiceId = it.next()
        case "--use-system-voice":
            args.useSystemVoice = true
        case "--rate":
            args.rate = Float(it.next() ?? "") ?? AVSpeechUtteranceDefaultSpeechRate
        case "--pitch":
            args.pitch = Float(it.next() ?? "") ?? 1.0
        case "--target-rate":
            args.targetRate = Int(it.next() ?? "24000") ?? 24000
        case "--help", "-h":
            printUsage()
            return nil
        default:
            // For backward compatibility: first positional arg is text, second is output path
            if args.text.isEmpty {
                args.text = arg
            } else if args.outputPath.isEmpty {
                args.outputPath = arg
            }
        }
    }

    // Validation
    if args.text.isEmpty {
        fputs("Error: --text is required\n", stderr)
        printUsage()
        return nil
    }

    if args.mode == .file && args.outputPath.isEmpty {
        fputs("Error: output path required for file mode\n", stderr)
        printUsage()
        return nil
    }

    return args
}

func printUsage() {
    let usage = """
    Siri TTS Sidecar - Native macOS Text-to-Speech

    USAGE:
      File mode:    siri-tts "text" output.wav [voice-id]
      Stream mode:  siri-tts --stream-pcm --text "text" [options]

    OPTIONS:
      --stream-pcm               Stream raw PCM to stdout (for Pipecat)
      --text "..."               Text to synthesize (required)
      --lang LANG                Language code (e.g., en-US, it-IT, es-ES)
      --voice-id ID              Specific voice identifier
      --use-system-voice        Use macOS Accessibility default voice (Spoken Content/VoiceOver)
      --rate FLOAT               Speech rate 0.0-1.0 (default: normal)
      --pitch FLOAT              Pitch multiplier (default: 1.0)
      --target-rate RATE         Target sample rate: 16000 or 24000 (default: 24000)

    EXAMPLES:
      # File mode (backward compatible)
      siri-tts "Hello" output.wav

      # Streaming mode for Pipecat
      siri-tts --stream-pcm --text "Hello from Siri" --lang en-US --target-rate 16000

      # Italian with voice customization
      siri-tts --stream-pcm --text "Ciao!" --lang it-IT --rate 0.52 --pitch 1.1
    """
    fputs(usage, stderr)
    fputs("\n", stderr)
}

// MARK: - Voice Selection

func selectVoice(args: Args) -> AVSpeechSynthesisVoice? {
    // Priority: 1) Explicit voice ID, 2) System default, 3) Language, 4) Ava
    if let vid = args.voiceId, let voice = AVSpeechSynthesisVoice(identifier: vid) {
        return voice
    }

    if args.useSystemVoice {
        if let systemId = systemDefaultVoiceIdentifier() ?? voiceOverDefaultVoiceIdentifier(),
           let v = AVSpeechSynthesisVoice(identifier: systemId) {
            return v
        }
    }

    if let lang = args.language {
        // Try enhanced voice first, then compact
        if let voice = AVSpeechSynthesisVoice(language: lang) {
            return voice
        }
    }

    // Default: Ava (US English, enhanced if available)
    if let ava = AVSpeechSynthesisVoice(identifier: "com.apple.voice.enhanced.en-US.Ava") {
        return ava
    } else if let ava = AVSpeechSynthesisVoice(identifier: "com.apple.voice.compact.en-US.Ava") {
        return ava
    }

    return nil
}

// Attempt to read the default voice from Spoken Content preferences
func systemDefaultVoiceIdentifier() -> String? {
    if let domain = UserDefaults.standard.persistentDomain(forName: "com.apple.speech.voice.prefs") as? [String: Any] {
        if let id = domain["SelectedVoiceIdentifier"] as? String { return id }
        if let selected = domain["SpokenContentSelectedVoiceIdentifier"] as? String { return selected }
        if let dict = domain["SelectedVoice"] as? [String: Any], let id = dict["VoiceIdentifier"] as? String { return id }
        if let name = domain["SelectedVoiceName"] as? String {
            for v in AVSpeechSynthesisVoice.speechVoices() {
                if v.name == name { return v.identifier }
            }
        }
    }
    return nil
}

// Attempt to read VoiceOver default voice
func voiceOverDefaultVoiceIdentifier() -> String? {
    if let domain = UserDefaults.standard.persistentDomain(forName: "com.apple.VoiceOver4/default") as? [String: Any] {
        if let attrs = domain["com.apple.speech.synthesis.voice-attributes"] as? [String: Any],
           let id = attrs["VoiceIdentifier"] as? String {
            return id
        }
        if let id = domain["VoiceIdentifier"] as? String { return id }
    }
    return nil
}

// MARK: - Audio Resampling

func resample(pcm: UnsafePointer<Int16>, frameCount: Int, fromRate: Double, toRate: Int) -> [Int16] {
    if Int(fromRate) == toRate {
        return Array(UnsafeBufferPointer(start: pcm, count: frameCount))
    }

    // Simple linear interpolation resampling
    let ratio = fromRate / Double(toRate)
    let outFrameCount = Int(Double(frameCount) / ratio)
    var output = [Int16](repeating: 0, count: outFrameCount)

    for i in 0..<outFrameCount {
        let srcIdx = Double(i) * ratio
        let idx0 = Int(srcIdx)
        let idx1 = min(idx0 + 1, frameCount - 1)
        let frac = srcIdx - Double(idx0)

        let val0 = Double(pcm[idx0])
        let val1 = Double(pcm[idx1])
        output[i] = Int16(val0 + (val1 - val0) * frac)
    }

    return output
}

// MARK: - WAV File Mode

func writeWAVHeader(_ fh: FileHandle, sampleRate: Double, channels: UInt16) {
    func u32(_ v: UInt32) -> Data {
        var value = v.littleEndian
        return Data(bytes: &value, count: MemoryLayout<UInt32>.size)
    }
    func u16(_ v: UInt16) -> Data {
        var value = v.littleEndian
        return Data(bytes: &value, count: MemoryLayout<UInt16>.size)
    }

    var hdr = Data()
    hdr.append("RIFF".data(using: .ascii)!)
    hdr.append(u32(0))                             // chunk size (to patch)
    hdr.append("WAVEfmt ".data(using: .ascii)!)
    hdr.append(u32(16))                            // PCM fmt chunk size
    hdr.append(u16(1))                             // PCM
    hdr.append(u16(channels))                      // channels
    hdr.append(u32(UInt32(sampleRate)))            // sample rate
    hdr.append(u32(UInt32(sampleRate) * UInt32(channels) * 2)) // byte rate
    hdr.append(u16(channels * 2))                  // block align
    hdr.append(u16(16))                            // bits per sample
    hdr.append("data".data(using: .ascii)!)
    hdr.append(u32(0))                             // data size (to patch)
    try! fh.write(contentsOf: hdr)
}

func runFileMode(args: Args) -> Int32 {
    let synth = AVSpeechSynthesizer()
    let utt = AVSpeechUtterance(string: args.text)

    utt.voice = selectVoice(args: args)
    utt.rate = args.rate
    utt.pitchMultiplier = args.pitch
    utt.volume = 1.0

    guard FileManager.default.createFile(atPath: args.outputPath, contents: nil),
          let fh = FileHandle(forWritingAtPath: args.outputPath) else {
        fputs("Error: Cannot create output file: \(args.outputPath)\n", stderr)
        return 1
    }

    var totalBytes: UInt64 = 0
    var wroteHeader = false
    var sampleRate: Double = Double(args.targetRate)
    let channels: UInt16 = 1

    let sem = DispatchSemaphore(value: 0)

    synth.write(utt) { buf in
        if let pcm = buf as? AVAudioPCMBuffer, pcm.frameLength > 0 {
            let fmt = pcm.format
            sampleRate = fmt.sampleRate

            if !wroteHeader {
                writeWAVHeader(fh, sampleRate: sampleRate, channels: channels)
                wroteHeader = true
            }

            let frameLen = Int(pcm.frameLength)

            // Handle Int16 PCM data
            if let ch0 = pcm.int16ChannelData?.pointee {
                let resampled = resample(pcm: ch0, frameCount: frameLen,
                                        fromRate: sampleRate, toRate: args.targetRate)
                let data = Data(bytes: resampled, count: resampled.count * 2)
                try? fh.write(contentsOf: data)
                totalBytes += UInt64(data.count)
            }
            // Handle Float32 PCM data - convert to Int16
            else if let ch0f = pcm.floatChannelData?.pointee {
                var tmpFloat = [Float](repeating: 0, count: frameLen)
                var scale: Float = 32767.0
                vDSP_vsmul(ch0f, 1, &scale, &tmpFloat, 1, vDSP_Length(frameLen))

                let tmpInt16 = tmpFloat.map { Int16(max(-32768, min(32767, $0))) }
                let resampled = resample(pcm: tmpInt16, frameCount: frameLen,
                                        fromRate: sampleRate, toRate: args.targetRate)
                let data = Data(bytes: resampled, count: resampled.count * 2)
                try? fh.write(contentsOf: data)
                totalBytes += UInt64(data.count)
            }
        } else {
            // Synthesis complete - patch WAV header with final sizes
            do {
                var chunkSize = UInt32(totalBytes + 36).littleEndian
                var dataSize = UInt32(totalBytes).littleEndian

                try fh.seek(toOffset: 4)
                try fh.write(contentsOf: Data(bytes: &chunkSize, count: 4))
                try fh.seek(toOffset: 40)
                try fh.write(contentsOf: Data(bytes: &dataSize, count: 4))
                try fh.close()
            } catch {
                fputs("Error writing WAV header: \(error)\n", stderr)
            }
            sem.signal()
        }
    }

    // Process RunLoop to allow callbacks
    let deadline = Date().addingTimeInterval(30)
    var completed = false

    while !completed && Date() < deadline {
        RunLoop.current.run(mode: .default, before: Date(timeIntervalSinceNow: 0.05))
        if sem.wait(timeout: .now()) == .success {
            completed = true
            break
        }
    }

    if !completed {
        fputs("Error: Synthesis timed out\n", stderr)
        return 1
    }

    print("ok \(args.outputPath)")
    return 0
}

// MARK: - PCM Streaming Mode

func runStreamMode(args: Args) -> Int32 {
    let synth = AVSpeechSynthesizer()
    let utt = AVSpeechUtterance(string: args.text)

    utt.voice = selectVoice(args: args)
    utt.rate = args.rate
    utt.pitchMultiplier = args.pitch
    utt.volume = 1.0

    let sem = DispatchSemaphore(value: 0)

    // Synthesis MUST run on main thread for RunLoop processing
    synth.write(utt) { buf in
        if let pcm = buf as? AVAudioPCMBuffer, pcm.frameLength > 0 {
            let fmt = pcm.format
            let frameLen = Int(pcm.frameLength)
            var pcm16: [Int16] = []

            // Convert to Int16 if needed
            if let ch0 = pcm.int16ChannelData?.pointee {
                pcm16 = Array(UnsafeBufferPointer(start: ch0, count: frameLen))
            } else if let ch0f = pcm.floatChannelData?.pointee {
                var tmpFloat = [Float](repeating: 0, count: frameLen)
                var scale: Float = 32767.0
                vDSP_vsmul(ch0f, 1, &scale, &tmpFloat, 1, vDSP_Length(frameLen))
                pcm16 = tmpFloat.map { Int16(max(-32768, min(32767, $0))) }
            }

            // Resample if needed
            let resampled = resample(pcm: pcm16, frameCount: pcm16.count,
                                    fromRate: fmt.sampleRate, toRate: args.targetRate)

            // Write raw PCM to stdout
            let data = Data(bytes: resampled, count: resampled.count * 2)
            data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                _ = write(STDOUT_FILENO, ptr.baseAddress, data.count)
            }
            fflush(stdout)
        } else {
            // Synthesis complete
            sem.signal()
        }
    }

    // Process RunLoop to allow callbacks (30s timeout)
    let deadline = Date().addingTimeInterval(30)
    var completed = false

    while !completed && Date() < deadline {
        RunLoop.current.run(mode: .default, before: Date(timeIntervalSinceNow: 0.05))
        if sem.wait(timeout: .now()) == .success {
            completed = true
            break
        }
    }

    if !completed {
        fputs("Error: Synthesis timed out\n", stderr)
        return 1
    }

    return 0
}

// MARK: - Main Entry Point

guard let args = parseArgs() else {
    exit(2)
}

let exitCode: Int32
switch args.mode {
case .file:
    exitCode = runFileMode(args: args)
case .stream:
    exitCode = runStreamMode(args: args)
}

exit(exitCode)
