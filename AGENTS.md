# AGENTS.md - Essential Briefing for AI Agents

> **For human contributors:** See README.md for getting started guide.

---

## 🤖 THE 3-ENTITY WORKFLOW (CORE CONCEPT)

This project uses a 3-entity collaboration pattern between Human, Droid Assistant, and Droid Exec.

### Roles & Responsibilities

**Human (You)**
- **Reads:** PROJECT_STATUS.md or equivalent (current progress, blockers)
- **Role:** Sets direction, provides requirements, reviews final output, approves changes
- **Decides:** What to build, priorities, when work is complete

**Droid Assistant (AI in Chat)**
- **Reads:** AGENTS.md (this file) + PROJECT_STATUS.md
- **Role:** Plans work, writes specs in `tasks/`, delegates to Droid Exec, reviews output
- **NEVER:** Writes code directly - always delegates to Droid Exec
- **Does:** Planning, spec writing, code review, coordination

**Droid Exec (Autonomous Agent)**
- **Reads:** Task specs (`tasks/*.md`) + AGENTS.md
- **Role:** Executes specs autonomously, implements code with TDD (tests first)
- **Always:** Follows AGENTS.md conventions, writes tests before code
- **Does:** Coding, testing, implementation

### The Workflow Pattern

```
Human → Tells Droid Assistant what to build
   ↓
Droid Assistant → Creates spec in tasks/
   ↓
Droid Assistant → Commits spec FIRST
   ↓
Droid Assistant → Runs: droid exec --auto medium -f tasks/spec.md
   ↓
Droid Exec → Writes tests → Implements code → All tests pass
   ↓
Droid Assistant → Reviews output, tests deployment (if needed)
   ↓
Droid Assistant → Commits Droid Exec's work
   ↓
Human → Tests and provides feedback → Iteration continues
```

### Critical Rules

1. **TDD is MANDATORY** when writing code
   - Droid Exec writes tests FIRST, then implementation
   - For simple work (HTML/CSS, docs), TDD may not be needed - use judgment

2. **NEVER write code directly**
   - Droid Assistant creates specs and delegates to Droid Exec
   - If you catch yourself writing code, STOP and write a spec instead

3. **Always commit spec before delegating**
   - Keeps git history clean (spec → implementation → completion)
   - Easy to see what was requested vs what was produced

4. **API key required for Droid Exec**
   ```bash
   export FACTORY_API_KEY="your-key-from-https://app.factory.ai/settings/api-keys"
   ```

### Example Workflow (Real Conversation)

```
Human: I need a contact form with validation

Droid: I'll create a spec for this. Let me check the project structure first.
$ ls src/components/
# [output shown]

━━━ Step 1: Droid Creates Spec ━━━
Droid: Creating task spec with TDD requirements...
$ vim tasks/contact_form.md

# Part of tasks/contact_form.md:
# 
# Create contact form component with validation
# - Email, name, message fields
# - Validation: email format, required fields
# - Submit handler with loading state
# 
# TDD Approach:
# - Write tests FIRST in __tests__/ContactForm.test.tsx
# - test_validates_email_format()
# - test_prevents_submit_when_invalid()
# - test_shows_loading_state()
# - test_handles_submit_success()
# 
# Then implement ContactForm.tsx to make tests pass

━━━ Step 2: Droid Commits Spec ━━━
Droid: Committing spec before delegation...
$ git add tasks/contact_form.md
$ git commit -m "spec: add contact form with validation"

━━━ Step 3: Droid Delegates to Droid Exec ━━━
Droid: Delegating to Droid Exec...
$ droid exec --auto medium -f tasks/contact_form.md
# Droid Exec: Writes tests FIRST → Implements component → All tests pass ✓

━━━ Step 4: Droid Reviews Output ━━━
Droid: Reviewing what Droid Exec created...
$ cat src/components/ContactForm.tsx
$ npm test __tests__/ContactForm.test.tsx
# ✓ 4/4 tests passing

━━━ Step 5: Droid Commits Work ━━━
Droid: Tests pass, committing...
$ git add src/components/ContactForm.tsx __tests__/ContactForm.test.tsx
$ git commit -m "feat: add contact form with validation (via droid exec)

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"

━━━ Step 6: Human Tests ━━━
Human: Form works great! Can you add a character limit to the message field?

Droid: I'll update the spec and re-delegate to Droid Exec...
```

### Droid Exec Autonomy Levels

- **`--auto low`**: File operations only (create, edit, delete)
- **`--auto medium`**: File ops + commands (npm, pip, pytest, git commit)


**How to choose:**
- Need to run tests? → medium
- Need to install packages? → medium
- Just creating/editing files? → low


### Reinforcing the Workflow

**Pro Tip:** If Droid Assistant starts writing code directly or skips delegation steps, remind it:

> "Follow AGENTS.md - delegate to droid exec instead."

Droid learns from feedback and will correct course.

---

## 📁 Project Setup

[AI Agent: Insert your project structure, file naming conventions, and import patterns here]

Example:
```
src/
├── components/         # React components
├── services/          # Business logic
└── __tests__/         # Tests mirror src/

tasks/                 # Task specs for droid exec
```

---

## 🛠️ Core Commands

[AI Agent: Insert your project's essential commands here]

Example:
```bash
npm install           # Install dependencies
npm run dev          # Start dev server
npm test             # Run tests
npm run build        # Production build
```

---

## 🎨 Coding Conventions

[AI Agent: Insert your formatting, linting, and code style rules here]

Example:
- Formatter: Prettier
- Linter: ESLint
- TypeScript strict mode
- 100-char line limit

---

## 🧪 Testing Standards

[AI Agent: Insert your testing approach, coverage requirements, and test organization here]

Example:
- Minimum 80% coverage
- Tests mirror src/ structure
- Test naming: `component.test.ts`

---

## 🔑 Environment & Services

[AI Agent: Insert required environment variables and external service configurations here]

Example:
```bash
API_KEY="your-key"
DATABASE_URL="postgres://..."
```

---

## ⚠️ Project-Specific Gotchas

[AI Agent: Insert project-specific issues, workarounds, and important notes here]

Example:
- Empty repos: Create README first
- API rate limits: 5000 req/hour
- Bundle size: Keep under 1MB

---

## 🚀 Git Workflow

[AI Agent: Insert your branching strategy, commit message format, and merge policies here]

Example:
```bash
git checkout main
git pull
# make changes via droid exec
git commit -m "type: description"
git push
```

---

## 📚 Tech Stack

[AI Agent: Insert your framework, libraries, and tooling choices here]

Example:
- Frontend: React 18 + TypeScript
- Styling: TailwindCSS
- Testing: Vitest
- Deploy: Vercel

---

## 🔗 Resources

[AI Agent: Insert links to documentation, templates, and helpful resources here]

Example:
- Droid Exec Examples: `tasks/` directory
- Component Library: ui.shadcn.com
- API Docs: [link]

---

**Remember:** The 3-entity workflow (Human → Droid Assistant → Droid Exec) is the core pattern. When in doubt, refer back to the workflow diagram at the top.