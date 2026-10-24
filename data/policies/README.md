# Saudi privacy policies

11 privacy policies of apps and websites that operate in Saudi Arabia. They are the
test set for the Gap Analyser: each one is annotated by the team against the PDPL
requirements checklist.

| File | What it is |
|---|---|
| `sources.csv` | The list: id, name, sector, language, link, and how the text is collected |
| `<policy_id>.txt` | The policy text |
| `policies.csv` | `sources.csv` plus the download date, status and text length |

To download them again:

```
python -m src.policies.fetch_policies
```

`saudia.txt` and `alrajhi.txt` are copied by hand from the browser: the Saudia site blocks
scripts, and the Al Rajhi page loads its text with JavaScript.

**Second language (for the app).** The app shows a policy in the language of its interface.
`other_language.csv` lists each policy's other language (Arabic for 8, English for alinma and
stc; KFUPM has no Arabic policy page); `python -m src.policies.fetch_other_language` saves them in `ar/<policy_id>.txt` and
`en/<policy_id>.txt` and writes `other_language_fetched.csv`. A page that comes back in the
wrong language is not saved. Al Rajhi, Saudia and Jarir are copied by hand, then the script is run
again. These copies are not used in the evaluation, which stays on the labelled copies above.

**Copyright:** each policy belongs to its company. The texts are kept here only for
non-commercial, educational use in this project, with the source link and download date.
