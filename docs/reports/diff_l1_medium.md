### Extraction Diff Report

#### Example

- Text: The company will announce its quarterly earnings next week after the board meeting.
- Gold: [('company', 'announce', 'quarterly earnings')]
- A (YAML raw)  [1.000/1.000/1.000]: [('company', 'announce', 'quarterly earnings')]
- B (YAML judge)[1.000/1.000/1.000]: [('company', 'announce', 'quarterly earnings')]
- Kept: [('company', 'announce', 'quarterly earnings')]
- Dropped by judge: []

#### Example

- Text: Both the students and the teachers enjoyed the field trip to the science museum.
- Gold: [('students', 'enjoy', 'field trip'), ('teachers', 'enjoy', 'field trip')]
- A (YAML raw)  [0.000/0.000/0.000]: [('students', 'enjoy', 'field trip to the museum'), ('teachers', 'enjoy', 'field trip to the museum')]
- B (YAML judge)[0.000/0.000/0.000]: []
- Kept: []
- Dropped by judge: [('students', 'enjoy', 'field trip to the museum'), ('teachers', 'enjoy', 'field trip to the museum')]

#### Example

- Text: The chef prepared the salad and the main course for the distinguished guests.
- Gold: [('chef', 'prepare', 'salad'), ('chef', 'prepare', 'main course')]
- A (YAML raw)  [0.000/0.000/0.000]: [('chef', 'prepare', 'main course for distinguished guests'), ('chef', 'prepare', 'salad for distinguished guests')]
- B (YAML judge)[0.000/0.000/0.000]: []
- Kept: []
- Dropped by judge: [('chef', 'prepare', 'main course for distinguished guests'), ('chef', 'prepare', 'salad for distinguished guests')]

#### Example

- Text: The researchers have been studying the effects of climate change on marine ecosystems for several years.
- Gold: [('researchers', 'study', 'effects of climate change on marine ecosystems')]
- A (YAML raw)  [0.000/0.000/0.000]: [('researchers', 'study', 'effects of change on ecosystems for years'), ('change', 'affect', 'ecosystems')]
- B (YAML judge)[0.000/0.000/0.000]: [('change', 'affect', 'ecosystems')]
- Kept: [('change', 'affect', 'ecosystems')]
- Dropped by judge: [('researchers', 'study', 'effects of change on ecosystems for years')]

#### Example

- Text: The committee will review and approve the budget proposal before the end of the month.
- Gold: [('committee', 'review', 'budget proposal'), ('committee', 'approve', 'budget proposal')]
- A (YAML raw)  [0.500/0.500/0.500]: [('committee', 'review', ''), ('committee', 'approve', 'budget proposal')]
- B (YAML judge)[1.000/0.500/0.667]: [('committee', 'approve', 'budget proposal')]
- Kept: [('committee', 'approve', 'budget proposal')]
- Dropped by judge: [('committee', 'review', '')]

#### Example

- Text: The students turned in their assignments and left the classroom quietly.
- Gold: [('students', 'turn_in', 'assignments'), ('students', 'leave', 'classroom')]
- A (YAML raw)  [0.500/0.500/0.500]: [('students', 'turn_in', 'assignments'), ('students', 'leave', 'assignments')]
- B (YAML judge)[0.500/0.500/0.500]: [('students', 'turn_in', 'assignments'), ('students', 'leave', 'assignments')]
- Kept: [('students', 'leave', 'assignments'), ('students', 'turn_in', 'assignments')]
- Dropped by judge: []

#### Example

- Text: The new policy has been implemented across all departments to improve efficiency.
- Gold: [('policy', 'implement', 'across departments')]
- A (YAML raw)  [0.000/0.000/0.000]: [('policy', 'implement', 'departments')]
- B (YAML judge)[0.000/0.000/0.000]: [('policy', 'implement', 'departments')]
- Kept: [('policy', 'implement', 'departments')]
- Dropped by judge: []

#### Example

- Text: The professor explained the complex theory with several examples and diagrams.
- Gold: [('professor', 'explain', 'theory')]
- A (YAML raw)  [0.000/0.000/0.000]: [('professor', 'explain', 'complex theory')]
- B (YAML judge)[0.000/0.000/0.000]: [('professor', 'explain', 'complex theory')]
- Kept: [('professor', 'explain', 'complex theory')]
- Dropped by judge: []

#### Example

- Text: The team has been working on the project since January and expects to complete it by December.
- Gold: [('team', 'work_on', 'project'), ('team', 'expect', 'complete it')]
- A (YAML raw)  [0.000/0.000/0.000]: [('team', 'work_because_of', 'expect'), ('team', 'work_on', 'on project'), ('team', 'expect', 'on project')]
- B (YAML judge)[0.000/0.000/0.000]: [('team', 'work_because_of', 'expect'), ('team', 'work_on', 'on project'), ('team', 'expect', 'on project')]
- Kept: [('team', 'expect', 'on project'), ('team', 'work_because_of', 'expect'), ('team', 'work_on', 'on project')]
- Dropped by judge: []

#### Example

- Text: The manager asked the employees to submit their reports by Friday and attend the meeting on Monday.
- Gold: [('manager', 'ask', 'employees'), ('employees', 'submit', 'reports'), ('employees', 'attend', 'meeting')]
- A (YAML raw)  [0.500/0.333/0.400]: [('manager', 'submit', 'employees'), ('manager', 'ask', 'employees')]
- B (YAML judge)[0.500/0.333/0.400]: [('manager', 'submit', 'employees'), ('manager', 'ask', 'employees')]
- Kept: [('manager', 'ask', 'employees'), ('manager', 'submit', 'employees')]
- Dropped by judge: []

### Aggregate
- YAML raw: P=0.222 R=0.235 F1=0.229
- YAML judge: P=0.333 R=0.235 F1=0.276