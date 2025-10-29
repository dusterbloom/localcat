#!/usr/bin/env node
/**
 * Verification script for the duplicate text fix
 *
 * This script verifies that the whitespace normalization logic
 * correctly catches the duplicate text bug scenario.
 */

// Helper function from the fix (improved version)
const normalizeWhitespace = (text) => {
  return text
    .replace(/\s+/g, ' ')  // Collapse multiple spaces
    .replace(/\s*([,;:.!?])\s*/g, '$1')  // Remove spaces around punctuation
    .trim();
};

// Test data from actual server logs
const testCases = [
  {
    name: 'Exact duplicate from logs',
    text1: 'I can remember details from our conversation and previous interactions. You can think of me as having a short-term memory, able to recall the latest 3+ turns of our chat.',
    text2: 'I can remember details from our conversation and previous interactions. You can think of me as having a short-term memory,able to recall the latest 3+ turns of our chat.',
    shouldMatch: true
  },
  {
    name: 'Fragment vs full text',
    text1: 'You can think of me as having a short-term memory,',
    text2: 'You can think of me as having a short-term memory, able to recall the latest 3+ turns of our chat.',
    shouldMatch: false, // Not exact, but text2 should contain text1
    shouldContain: true
  },
  {
    name: 'Multiple spaces normalization',
    text1: 'You  can   think    of     me',
    text2: 'You can think of me',
    shouldMatch: true
  },
  {
    name: 'Different texts',
    text1: 'Hello, how are you?',
    text2: 'Goodbye, see you later!',
    shouldMatch: false,
    shouldContain: false
  }
];

console.log('🧪 Testing Duplicate Detection Fix\n');
console.log('=' .repeat(80));

let passed = 0;
let failed = 0;

testCases.forEach((testCase, index) => {
  console.log(`\nTest ${index + 1}: ${testCase.name}`);
  console.log('-'.repeat(80));

  const normalized1 = normalizeWhitespace(testCase.text1);
  const normalized2 = normalizeWhitespace(testCase.text2);

  console.log(`Text 1: "${testCase.text1}"`);
  console.log(`Text 2: "${testCase.text2}"`);
  console.log(`\nNormalized 1: "${normalized1}"`);
  console.log(`Normalized 2: "${normalized2}"`);

  const exactMatch = normalized1 === normalized2;
  const contains = normalized1.includes(normalized2) || normalized2.includes(normalized1);

  console.log(`\nExact match: ${exactMatch}`);
  console.log(`Contains: ${contains}`);

  let testPassed = true;

  if (testCase.shouldMatch !== undefined) {
    if (exactMatch === testCase.shouldMatch) {
      console.log(`✅ Exact match test passed`);
    } else {
      console.log(`❌ Exact match test failed (expected ${testCase.shouldMatch}, got ${exactMatch})`);
      testPassed = false;
    }
  }

  if (testCase.shouldContain !== undefined) {
    if (contains === testCase.shouldContain) {
      console.log(`✅ Contains test passed`);
    } else {
      console.log(`❌ Contains test failed (expected ${testCase.shouldContain}, got ${contains})`);
      testPassed = false;
    }
  }

  if (testPassed) {
    passed++;
  } else {
    failed++;
  }
});

console.log('\n' + '='.repeat(80));
console.log(`\nResults: ${passed} passed, ${failed} failed`);

if (failed === 0) {
  console.log('\n✅ All tests passed! The fix correctly detects duplicates.');
  process.exit(0);
} else {
  console.log('\n❌ Some tests failed. Please review the implementation.');
  process.exit(1);
}
