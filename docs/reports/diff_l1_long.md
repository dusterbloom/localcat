### Extraction Diff Report

#### Example

- Text: Despite numerous delays and budget constraints, the committee eventually approved the revised proposal for expanding the network infrastructure across all regional offices.
- Gold: [('committee', 'approve', 'revised proposal')]
- A (YAML raw)  [1.000/1.000/1.000]: [('committee', 'approve', 'revised proposal')]
- B (YAML judge)[1.000/1.000/1.000]: [('committee', 'approve', 'revised proposal')]
- Kept: [('committee', 'approve', 'revised proposal')]
- Dropped by judge: []

#### Example

- Text: The research team, which consists of scientists from five different countries, has been collecting and analyzing data from various sources to develop a comprehensive model of climate change effects on agricultural productivity.
- Gold: [('research team', 'collect', 'data'), ('research team', 'analyze', 'data'), ('research team', 'develop', 'model')]
- A (YAML raw)  [0.000/0.000/0.000]: [('team', 'collect', 'scientists'), ('which', 'consist_of', 'scientists')]
- B (YAML judge)[0.000/0.000/0.000]: [('team', 'analyze', 'scientists'), ('team', 'collect', 'scientists')]
- Kept: [('team', 'collect', 'scientists')]
- Dropped by judge: [('which', 'consist_of', 'scientists')]
- Added by judge: [('team', 'analyze', 'scientists')]

#### Example

- Text: After reviewing the financial reports and consulting with the legal department, the board of directors decided to postpone the merger until the regulatory approval process could be completed without any complications.
- Gold: [('board', 'decide', 'postpone merger'), ('board', 'postpone', 'merger')]
- A (YAML raw)  [0.500/0.500/0.500]: [('board', 'decide', 'postpone merger'), ('board', 'postpone', 'postpone merger')]
- B (YAML judge)[0.500/0.500/0.500]: [('board', 'decide', 'postpone merger'), ('board', 'postpone', 'postpone merger')]
- Kept: [('board', 'decide', 'postpone merger'), ('board', 'postpone', 'postpone merger')]
- Dropped by judge: []

#### Example

- Text: The new educational program, designed specifically for students with learning disabilities, incorporates various teaching methods and technological tools to enhance comprehension and retention of complex mathematical concepts.
- Gold: [('program', 'incorporate', 'methods'), ('program', 'incorporate', 'tools')]
- A (YAML raw)  [0.000/0.000/0.000]: [('program', 'incorporate', 'teaching methods')]
- B (YAML judge)[0.000/0.000/0.000]: [('program', 'incorporate', 'teaching methods')]
- Kept: [('program', 'incorporate', 'teaching methods')]
- Dropped by judge: []

#### Example

- Text: The international conference, attended by delegates from over fifty countries, focused on discussing innovative approaches to renewable energy and establishing collaborative frameworks for research and development in sustainable technologies.
- Gold: [('conference', 'focus_on', 'approaches'), ('conference', 'establish', 'frameworks')]
- A (YAML raw)  [0.000/0.000/0.000]: [('conference', 'focus', '')]
- B (YAML judge)[0.000/0.000/0.000]: []
- Kept: []
- Dropped by judge: [('conference', 'focus', '')]

#### Example

- Text: The government has implemented stricter regulations on industrial emissions and invested heavily in green technologies, which according to environmental experts will significantly reduce air pollution in urban areas within the next decade.
- Gold: [('government', 'implement', 'regulations'), ('government', 'invest', 'technologies'), ('regulations', 'reduce', 'pollution')]
- A (YAML raw)  [0.000/0.000/0.000]: [('government', 'invest', 'green technologies'), ('which', 'reduce', 'green technologies')]
- B (YAML judge)[0.000/0.000/0.000]: [('government', 'invest', 'green technologies'), ('which', 'reduce', 'green technologies')]
- Kept: [('government', 'invest', 'green technologies'), ('which', 'reduce', 'green technologies')]
- Dropped by judge: []

#### Example

- Text: The marketing team, after conducting extensive market research and analyzing consumer behavior patterns, proposed a comprehensive strategy that includes both traditional advertising methods and innovative digital campaigns to increase brand awareness among younger demographics.
- Gold: [('team', 'propose', 'strategy'), ('strategy', 'include', 'methods'), ('strategy', 'include', 'campaigns')]
- A (YAML raw)  [0.500/0.333/0.400]: [('younger demographics', 'include', 'strategy'), ('team', 'propose', 'strategy')]
- B (YAML judge)[0.500/0.333/0.400]: [('younger demographics', 'include', 'strategy'), ('team', 'propose', 'strategy')]
- Kept: [('team', 'propose', 'strategy'), ('younger demographics', 'include', 'strategy')]
- Dropped by judge: []

#### Example

- Text: Despite facing numerous challenges including supply chain disruptions and increased competition, the company managed to achieve record profits by diversifying its product line and expanding into emerging markets through strategic partnerships with local distributors.
- Gold: [('company', 'manage', 'achieve profits'), ('company', 'diversify', 'product line'), ('company', 'expand', 'into markets')]
- A (YAML raw)  [0.000/0.000/0.000]: [('company', 'manage', 'achieve record profits'), ('company', 'achieve', 'achieve record profits')]
- B (YAML judge)[0.000/0.000/0.000]: []
- Kept: []
- Dropped by judge: [('company', 'achieve', 'achieve record profits'), ('company', 'manage', 'achieve record profits')]

### Aggregate
- YAML raw: P=0.231 R=0.158 F1=0.187
- YAML judge: P=0.300 R=0.158 F1=0.207