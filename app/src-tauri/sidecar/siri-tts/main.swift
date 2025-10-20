// Siri TTS sidecar for LocalCat - uses AVSpeechSynthesizer for offline, native macOS voices
import AVFoundation
import Accelerate

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

let args = CommandLine.arguments
guard args.count >= 3 else {
    fputs("Usage: siri-tts \"text\" out.wav [voice-id]\n", stderr)
    fputs("Example: siri-tts \"Hello from LocalCat\" /tmp/hello.wav\n", stderr)
    fputs("         siri-tts \"Hello\" out.wav com.apple.voice.enhanced.en-US.Ava\n", stderr)
    exit(2)
}

let text = args[1]
let outPath = args[2]
let voiceId = (args.count >= 4) ? args[3] : nil

let synth = AVSpeechSynthesizer()
let utt = AVSpeechUtterance(string: text)

// If voice ID provided, use it; otherwise use default Ava voice
if let vid = voiceId, let v = AVSpeechSynthesisVoice(identifier: vid) {
    utt.voice = v
} else {
    // Default to Ava (enhanced US English female voice)
    if let ava = AVSpeechSynthesisVoice(identifier: "com.apple.voice.enhanced.en-US.Ava") {
        utt.voice = ava
    } else if let ava = AVSpeechSynthesisVoice(identifier: "com.apple.voice.compact.en-US.Ava") {
        utt.voice = ava
    }
}

// Tune speech parameters
utt.rate = AVSpeechUtteranceDefaultSpeechRate
utt.pitchMultiplier = 1.0
utt.volume = 1.0

// Create output file
guard FileManager.default.createFile(atPath: outPath, contents: nil),
      let fh = FileHandle(forWritingAtPath: outPath) else {
    fputs("Error: Cannot create output file: \(outPath)\n", stderr)
    exit(1)
}

var totalBytes: UInt64 = 0
var wroteHeader = false
var sampleRate: Double = 24000  // Default to 24kHz for Pipecat
var channels: UInt16 = 1

let sem = DispatchSemaphore(value: 0)

var bufferCount = 0
synth.write(utt) { buf in
    bufferCount += 1
    fputs("[DEBUG] Callback #\(bufferCount): buf=\(String(describing: buf))\n", stderr)

    if let pcm = buf as? AVAudioPCMBuffer, pcm.frameLength > 0 {
        // Process PCM audio buffer with actual data
        fputs("[DEBUG] Processing PCM buffer with \(pcm.frameLength) frames\n", stderr)
        let fmt = pcm.format
        sampleRate = fmt.sampleRate
        channels = UInt16(fmt.channelCount)

        if !wroteHeader {
            writeWAVHeader(fh, sampleRate: sampleRate, channels: channels)
            wroteHeader = true
            fputs("[DEBUG] Wrote WAV header\n", stderr)
        }

        let frameLen = Int(pcm.frameLength)

        // Handle Int16 PCM data
        if let ch0 = pcm.int16ChannelData?.pointee {
            let data = Data(bytes: ch0, count: frameLen * 2)
            try? fh.write(contentsOf: data)
            totalBytes += UInt64(data.count)
        }
        // Handle Float32 PCM data - convert to Int16
        else if let ch0f = pcm.floatChannelData?.pointee {
            // Convert Float32 [-1,1] -> Int16
            var tmpFloat = [Float](repeating: 0, count: frameLen)
            var scale: Float = 32767.0
            vDSP_vsmul(ch0f, 1, &scale, &tmpFloat, 1, vDSP_Length(frameLen))

            let tmp = tmpFloat.map { Int16(max(-32768, min(32767, $0))) }
            let data = Data(bytes: tmp, count: frameLen * 2)
            try? fh.write(contentsOf: data)
            totalBytes += UInt64(data.count)
        }
    } else {
        // Synthesis complete (nil buffer) - patch WAV header with final sizes
        fputs("[DEBUG] Received non-PCM buffer (completion signal), finalizing...\n", stderr)
        do {
            var chunkSize = UInt32(totalBytes + 36).littleEndian
            var dataSize = UInt32(totalBytes).littleEndian

            try fh.seek(toOffset: 4)
            try fh.write(contentsOf: Data(bytes: &chunkSize, count: 4))
            try fh.seek(toOffset: 40)
            try fh.write(contentsOf: Data(bytes: &dataSize, count: 4))
            try fh.close()
            fputs("[DEBUG] Finalized WAV file, signaling semaphore\n", stderr)
        } catch {
            fputs("Error writing WAV header: \(error)\n", stderr)
        }
        sem.signal()
    }
}

// Process RunLoop to allow callbacks to fire in CLI context
fputs("[DEBUG] Starting RunLoop processing\n", stderr)
let deadline = Date().addingTimeInterval(30)
var completed = false

// Continuously poll RunLoop while waiting for synthesis to complete
while !completed && Date() < deadline {
    // Process one cycle of the RunLoop (50ms intervals)
    RunLoop.current.run(mode: .default, before: Date(timeIntervalSinceNow: 0.05))

    // Non-blocking check if semaphore was signaled
    if sem.wait(timeout: .now()) == .success {
        fputs("[DEBUG] Semaphore signaled, synthesis complete!\n", stderr)
        completed = true
        break
    }
}

if !completed {
    fputs("[ERROR] Synthesis timed out after 30 seconds\n", stderr)
    exit(1)
}

print("ok \(outPath)")
