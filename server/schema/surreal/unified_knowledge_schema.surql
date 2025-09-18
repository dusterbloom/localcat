-- Unified SurrealDB schema for HotMem

-- Entities
DEFINE TABLE entity SCHEMAFULL;
DEFINE FIELD kind        ON entity TYPE string ASSERT $value != NONE;
DEFINE FIELD name        ON entity TYPE string ASSERT string::len($value) > 0;
DEFINE FIELD created_at  ON entity TYPE datetime DEFAULT time::now();
DEFINE INDEX entity_name ON entity COLUMNS name;

-- Knowledge edges as native graph relations
DEFINE TABLE edge TYPE RELATION FROM entity TO entity SCHEMAFULL;
DEFINE FIELD predicate   ON edge   TYPE string ASSERT $value != NONE;
DEFINE FIELD confidence  ON edge   TYPE number ASSERT $value >= 0.0 AND $value <= 1.0;
DEFINE FIELD ts          ON edge   TYPE datetime DEFAULT time::now();
DEFINE FIELD session_id  ON edge   TYPE option<string>;
DEFINE FIELD props       ON edge   TYPE option<object>;
DEFINE INDEX edge_pred   ON edge   COLUMNS predicate, ts DESC;
DEFINE INDEX edge_sess   ON edge   COLUMNS session_id, ts DESC;

-- Aliases (canonicalization)
DEFINE TABLE alias SCHEMAFULL;
DEFINE FIELD alias       ON alias  TYPE string;
DEFINE FIELD canonical   ON alias  TYPE string;
DEFINE INDEX alias_key   ON alias  COLUMNS alias UNIQUE;

-- Mentions / Summaries (FTS)
DEFINE TABLE mention SCHEMAFULL;
DEFINE FIELD eid         ON mention TYPE string;            -- 'session:<id>' | 'summary:<id>' | 'entity:<id>'
DEFINE FIELD text        ON mention TYPE string;
DEFINE FIELD ts          ON mention TYPE datetime DEFAULT time::now();
DEFINE FIELD session_id  ON mention TYPE option<string>;
DEFINE FIELD turn_id     ON mention TYPE option<number>;

DEFINE ANALYZER english TOKENIZERS ascii, camel, punct, simple, lowercase FILTERS snowball(english);
DEFINE INDEX mention_fts ON TABLE mention FIELDS text SEARCH ANALYZER english;

-- Messages (optional: move SessionStore later)
DEFINE TABLE message SCHEMAFULL;
DEFINE FIELD role        ON message TYPE string ASSERT $value IN ['user','assistant','system'];
DEFINE FIELD content     ON message TYPE string ASSERT $value != NONE AND $value != '';
DEFINE FIELD session_id  ON message TYPE string;
DEFINE FIELD speaker_id  ON message TYPE string DEFAULT 'default_user';
DEFINE FIELD ts          ON message TYPE datetime DEFAULT time::now();
DEFINE INDEX msg_sess_ts ON message COLUMNS session_id, ts DESC;

-- Dev access (restrict in production)
DEFINE ACCESS local PERMISSIONS FULL;

