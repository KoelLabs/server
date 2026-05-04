# Phoneme Descriptions Schema

This is the desired cleaned schema for phoneme descriptions served to the app.
The server stores a superset of app-visible phonemes. App-visible/core phonemes
must have `primary: true`; additional server-only phonemes must have
`primary: false`.

## Required Fields

- `phoneme: string` - canonical phoneme identifier used by the app and server.
- `primary: boolean` - whether this is part of the app's primary phoneme library.
- `label: string` - friendly sound title.
- `plain_english: string` - short plain-English description.
- `description: string[]` - phonetic descriptors.
- `explanation: string` - coaching explanation.
- `primary_cue: string` - main articulation cue.
- `visual_cue: string` - visible mouth/face cue.
- `feel_check: string` - proprioceptive cue.
- `anchor_words: string[]` - short anchor word list.
- `minimal_pairs: [string, string][]` - default minimal pairs.
- `minimal_pairs_by_confusion: Record<string, [string, string][]>` - targeted
  minimal pairs keyed by confusion phoneme.
- `drill_ladder: { sound: string[]; syllables: string[]; words: string[]; phrases: string[] }`
  - practice ladder content.
- `examples: { word: string; phonetic_spelling: string }[]` - display examples.
- `phonetic_spelling: string` - visible pronunciation hint.
- `video: string` - feedback/demo video path or URL.

## Optional Fields

- `audio: string` - playable sample audio path or URL.
- `confused_phone: string` - default likely confusion phoneme.
- `sort_key: string` - app ordering hint for primary phonemes.
- `voicing_tag: string` - short display tag for paired/related sounds.
