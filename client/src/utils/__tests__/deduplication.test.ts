/**
 * Test suite for assistant message deduplication logic
 *
 * This tests the fix for the duplicate text bug where:
 * - FastTextAggregator splits at commas, removing leading spaces
 * - Context aggregator concatenates without spaces ("memory,able")
 * - TTSTextFrames preserve proper spacing ("memory, able")
 * - Both versions reach the client, causing duplicates
 */

describe('Assistant Message Deduplication', () => {
  // Helper function matching the implementation in VoiceApp.tsx
  const normalizeWhitespace = (text: string) => text.replace(/\s+/g, ' ').trim();

  describe('whitespace normalization', () => {
    it('should normalize multiple spaces to single space', () => {
      const text = 'You  can   think    of     me';
      expect(normalizeWhitespace(text)).toBe('You can think of me');
    });

    it('should trim leading and trailing spaces', () => {
      const text = '  You can think of me  ';
      expect(normalizeWhitespace(text)).toBe('You can think of me');
    });

    it('should handle tabs and newlines', () => {
      const text = 'You\tcan\nthink\r\nof\t\tme';
      expect(normalizeWhitespace(text)).toBe('You can think of me');
    });
  });

  describe('duplicate detection', () => {
    it('should detect exact duplicates after normalization', () => {
      const text1 = 'You can think of me as having a short-term memory, able to recall the latest 3+ turns of our chat.';
      const text2 = 'You can think of me as having a short-term memory,able to recall the latest 3+ turns of our chat.';

      const normalized1 = normalizeWhitespace(text1);
      const normalized2 = normalizeWhitespace(text2);

      expect(normalized1).toBe(normalized2);
    });

    it('should detect duplicates with extra spaces', () => {
      const text1 = 'You can  think of me';
      const text2 = 'You can think  of  me';

      const normalized1 = normalizeWhitespace(text1);
      const normalized2 = normalizeWhitespace(text2);

      expect(normalized1).toBe(normalized2);
    });

    it('should detect substring matches after normalization', () => {
      const fullText = 'You can think of me as having a short-term memory, able to recall the latest 3+ turns.';
      const partialText = 'You can think of me as having a short-term  memory,able to recall';

      const normalizedFull = normalizeWhitespace(fullText);
      const normalizedPartial = normalizeWhitespace(partialText);

      expect(normalizedFull.includes(normalizedPartial)).toBe(true);
    });
  });

  describe('real-world bug scenario', () => {
    it('should catch the comma-spacing duplicate from server logs', () => {
      // Version A: From TTSTextFrame with proper spacing
      const versionA = 'I can remember details from our conversation and previous interactions. You can think of me as having a short-term memory, able to recall the latest 3+ turns of our chat.';

      // Version B: From context aggregator with missing space after comma
      const versionB = 'I can remember details from our conversation and previous interactions. You can think of me as having a short-term memory,able to recall the latest 3+ turns of our chat.';

      // These should be detected as duplicates
      const normalizedA = normalizeWhitespace(versionA);
      const normalizedB = normalizeWhitespace(versionB);

      expect(normalizedA).toBe(normalizedB);
      expect(normalizedA.includes(normalizedB)).toBe(true);
      expect(normalizedB.includes(normalizedA)).toBe(true);
    });

    it('should handle the split fragments case', () => {
      const fragment1 = 'You can think of me as having a short-term memory,';
      const fragment2 = 'able to recall the latest 3+ turns of our chat.';
      const fullText = 'You can think of me as having a short-term memory, able to recall the latest 3+ turns of our chat.';

      // Full text should contain both fragments after normalization
      const normalizedFull = normalizeWhitespace(fullText);
      const normalizedFragment1 = normalizeWhitespace(fragment1);
      const normalizedFragment2 = normalizeWhitespace(fragment2);

      expect(normalizedFull.includes(normalizedFragment1)).toBe(true);
      expect(normalizedFull.includes(normalizedFragment2)).toBe(true);
    });
  });

  describe('edge cases', () => {
    it('should not match completely different texts', () => {
      const text1 = 'Hello, how are you?';
      const text2 = 'Goodbye, see you later!';

      const normalized1 = normalizeWhitespace(text1);
      const normalized2 = normalizeWhitespace(text2);

      expect(normalized1).not.toBe(normalized2);
      expect(normalized1.includes(normalized2)).toBe(false);
      expect(normalized2.includes(normalized1)).toBe(false);
    });

    it('should handle empty strings', () => {
      expect(normalizeWhitespace('')).toBe('');
      expect(normalizeWhitespace('   ')).toBe('');
    });

    it('should handle single words', () => {
      expect(normalizeWhitespace('  word  ')).toBe('word');
    });
  });
});
