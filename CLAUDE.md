# Coursework guidelines

NUS coursework. AI assistance is allowed here, but what gets submitted has to be
mine: my voice in the writing, my level in the code, and something I can explain
if a tutor asks about any line of it. Help me get there, don't hand me a finished
artifact I don't understand.

Follow whatever AI declaration rule the specific module has. If a module asks for
a declaration of AI use, remind me to fill it in honestly.

## No AI fingerprints

- No `Co-Authored-By: Claude` or any other co-author trailer in commits.
- No "Generated with Claude Code", "Made with AI", or similar in commit messages,
  READMEs, report footers, or acknowledgements.
- No mention of Claude, Anthropic, ChatGPT, or "the AI" in code comments,
  docstrings, or documentation.
- Commit messages: short and plain, no emoji, no conventional-commit ceremony
  unless the module asks for it. "fix join in q3", not
  "feat(query): implement optimized join strategy for question 3".

## Writing (reports, READMEs, answer sheets)

Characters first, since these survive copy-paste and are the fastest giveaway:

- No em dashes (—) and no en dashes used as punctuation. Use a comma, a colon,
  parentheses, or just split the sentence in two.
- Straight quotes (' and "), never curly ones.
- Three dots (...), never the single ellipsis character.
- Plain ASCII arrows (->), never →.
- No emoji anywhere, including section headings.

Words and phrases to avoid:

delve, leverage (as a verb), robust, seamless, comprehensive, crucial, pivotal,
testament, landscape, realm, underscore (as a verb), foster, myriad, intricate,
nuanced, holistic, "it's worth noting", "it's important to note", "at its core",
"plays a vital role", "in today's world".

Structural habits to avoid:

- The "not just X, but Y" construction, and its cousin "isn't merely X, it's Y".
- Rule-of-three lists everywhere. Real writing has twos and fours in it.
- Bolding the lead phrase of every bullet.
- Opening a section by restating the question that was asked.
- Closing with "In conclusion" or a summary paragraph that repeats what was just
  said. When a section is done, stop.
- Uniform paragraph length. Vary it. A one-sentence paragraph is fine.
- Turning everything into a nested bullet hierarchy. Most report sections should
  be plain prose, with maybe one list where a list genuinely helps.
- Hedging every claim ("may potentially", "could arguably"). Say the thing.

Register: write like a competent student explaining their own work, not like
product documentation. Contractions are fine. Short sentences are fine. Slightly
plain beats suspiciously polished.

## Code

Pitch it as idiomatic and correct but nothing flashy, roughly what someone who
has done the module readings would write on a good day.

- No premature abstraction. A script that runs once doesn't need a class
  hierarchy, a config object, or a `main()` guard unless the module's style uses
  one.
- No defensive error handling on homework. Skip try/except, input validation and
  logging setup unless the task actually asks for it.
- Comment sparingly, and only where a human would want a reminder of why
  something is done. Never a comment that restates the line below it.
- No banner comments (`# ===== LOADING DATA =====`) or decorative dividers.
- No docstring on every small function in a homework script.
- Plain variable names: `df`, `counts`, `top_sales`, not
  `aggregated_sales_dataframe_by_region`.
- Match the style already in the file and its siblings: indentation, casing,
  naming, quoting. Don't reformat code I already wrote.
- If the clean solution needs something the module probably hasn't covered, say
  so and offer the simpler version alongside it so I can choose.

### SQL

- Match the keyword casing and layout of the existing `.sql` files in that
  project folder instead of imposing a different house style.
- Prefer joins and subqueries the course has covered over window functions or
  CTEs, unless the question calls for them or I say the module taught them.
- Keep queries readable rather than golfed. One clause per line is fine.
- Don't add comment headers above every query beyond what the assignment
  template already has.

## How to work with me

- Explain the approach briefly alongside the code, so I can answer a follow-up
  question about why it works.
- Flag anything I would struggle to explain in a viva or a follow-up email.
- Ask before rewriting large chunks of something I already wrote.
- If I ask for something that would be over the line for a module, say so once,
  then do the version that isn't.

## Before I submit

Run these from the project folder and look at what comes back:

```bash
grep -rnP '[\x{2014}\x{2013}\x{2018}\x{2019}\x{201C}\x{201D}\x{2026}\x{2192}]' .
grep -rniE 'delve|leverag|seamless|robust|comprehensive|testament|in conclusion|worth noting' .
```

Then read it once more and ask whether any paragraph sounds like something I
couldn't say out loud in a tutorial. If it does, rewrite it.
