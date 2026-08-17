# Role Packages

A Role Package is the deployable character unit for the local robot Agent
Runtime. It keeps character behavior, model defaults, voice defaults, memory
policy, and capability permissions together without coupling them to DDS, ASR,
TTS, or a particular LLM server.

```text
roles/my-character/
├── role.json
├── prompt.md
└── lore/
    ├── core.md
    └── cards.jsonl
```

The manifest is JSON so loading requires only the Python standard library. It is
read and validated once during process startup. Only `prompt.md` becomes the
system message; manifest metadata does not consume model context tokens.

## Manifest version 1

```json
{
  "schema_version": 1,
  "id": "studio.my-character",
  "version": "1.0.0",
  "display_name": "My Character",
  "prompt": "prompt.md",
  "loop": "conversational",
  "model": {
    "max_tokens": 64,
    "temperature": 0.2,
    "thinking": false,
    "thinking_budget_tokens": -1
  },
  "voice": {
    "voice": "",
    "language": "",
    "instructions": ""
  },
  "memory": {
    "provider": "window",
    "max_turns": 4
  },
  "knowledge": {
    "provider": "keyword",
    "core": "lore/core.md",
    "cards": "lore/cards.jsonl",
    "max_cards": 2
  },
  "capabilities": {
    "allow": [],
    "deny": []
  }
}
```

`schema_version`, `id`, `version`, `display_name`, and `prompt` are required.
Package IDs use lowercase letters, digits, dots, underscores, and hyphens.
Prompt paths cannot escape the package directory.

Only `conversational` is implemented as a loop in version 1. This is an
intentional latency boundary: one user turn produces exactly one streaming LLM
request. Graph, OODA, planning, multi-agent, and autonomous tool loops are not
silently enabled.

Memory supports:

- `window`: retain at most `max_turns` completed user/assistant pairs in host
  memory;
- `none`: stateless requests.

Knowledge is separate from conversation memory. `core.md` contains a compact,
fixed character bible and is appended to the stable system prefix. A
`cards.jsonl` file contains query-specific facts:

```json
{"id":"friend","aliases":["friend","朋友","友達"],"facts":"A concise canonical fact."}
```

The `keyword` provider normalizes Unicode and matches multilingual aliases in
host memory. It injects at most `max_cards` facts into the existing request. It
does not load an embedding model, create a vector database, access the network,
or make a second LLM call. Cards are parsed and validated once at startup.
`provider: "none"` disables role knowledge.

Capability permissions are deny-by-default. An empty `allow` array exposes no
tools. A name in `deny` always wins, including when `allow` contains `"*"`.
Permission alone does not activate a tool: the process must separately register
a matching `CapabilityProvider`. This separates character authorization from
robot-specific implementation.

CLI flags override package defaults for one deployment. For example:

```bash
scripts/run-local-voice-agent.sh \
  --role-package roles/olaf \
  --history-turns 2 \
  --no-thinking
```

The legacy `--role-file` prompt option remains available for compatibility, but
new reusable characters should use Role Packages.
