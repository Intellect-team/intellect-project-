# AI First-Aid Guide — MVP Build Plan

Camera → classify injury → generate first-aid steps → show on screen, with an emergency-services disclaimer always visible.

Since you already know Flutter, FastAPI, and Supabase (from WAVE), this plan builds on that stack instead of introducing new tools — it should get you to a working demo fastest.

---

## 1. Architecture

```
[Phone Camera] → [Flutter App] → (photo) → [FastAPI Backend]
                                                 │
                                    ┌────────────┴────────────┐
                              [Vision step]              [LLM step]
                         (classify: burn/cut/            (Claude API:
                          choking-sign/normal)             turn label →
                                                            plain-language
                                                            first-aid steps)
                                                 │
                                        [JSON response]
                                                 │
                                     [Flutter renders steps
                                      + "Call emergency
                                      services" banner]
```

Two-stage pipeline, not one model doing everything:
1. **Vision model** → outputs a label (burn / cut / choking-sign / normal) + confidence.
2. **LLM (Claude)** → takes that label and writes the plain-language steps. This is the part you can build and iterate on fastest, and it's also where your disclaimer text lives (hard-coded, never model-generated, so it can't be dropped by a bad completion).

Keep these decoupled. If the vision model misfires on stage, you can still fall back to letting the presenter tap the correct label manually and the LLM step still works — a safety net for a live demo.

---

## 2. Data prep (do this first, it's the long pole)

For each dataset:

1. **Download & dedupe.** For the burn dataset: run the filter script from its README to drop the 214 orphan `.txt` files, and diff `archive.zip` contents against the loose `imgN.jpg/txt` files so you don't double-count. For the MiliTech cuts dataset and ChokeDetection: export directly from Roboflow in YOLO format (Roboflow does the train/valid/test split for you).
2. **Collapse to your actual classes.** You don't need 10+ classes. Pick one flat label set, e.g.:
   `burn_mild, burn_severe, cut, choking_sign, normal`
   Merge first/second-degree burns into `burn_mild`, third-degree into `burn_severe`. Drop chronic-wound classes (diabetic/pressure/venous/ulcer) entirely — not in scope.
3. **Re-label programmatically**, not by hand — write a small script that remaps each source dataset's class IDs into your unified label map and rewrites the YOLO `.txt` files. This is mechanical, not manual annotation, since all three sources already come pre-labeled.
4. **Balance check.** After merging, count images per class. Your choking set (~1,200) and cuts set (~6,800 pre-filter) will dwarf the burn set (~1,200) once irrelevant classes are dropped — expect burn_severe (third-degree, 355 boxes originally) to be your thinnest class. Note it, don't panic — see §4.
5. **Split** 80/10/10 train/val/test, stratified by class so the small class isn't wiped out of validation.
6. **Sanity-check visually.** Render ~30 random images per class with their boxes drawn on (a 10-line script with OpenCV/PIL) and eyeball them before training anything — mislabeled boxes are the #1 silent killer of small datasets.

---

## 3. The model decision — build vs. prompt

Given the timeline (hackathon/demo, not production), I'd genuinely recommend **not** fine-tuning your own detector for the first version. Reasons:
- Your combined, cleaned dataset is still only a few thousand images across imbalanced classes — a from-scratch or lightly fine-tuned YOLO model trained in a few days will be shaky, and a shaky model on stage is worse than a simple one.
- A vision-capable LLM (e.g., Claude with vision, or GPT-4V) prompted with a few labeled reference images (few-shot) plus your target photo can classify "burn / cut / choking-sign / normal" competently out of the box, with zero training time.

**Recommended MVP path:**
- **Stage classification with a prompted vision-language model**, not a trained CNN. One API call: send the photo + a short instruction + 2-3 reference examples per class → get back a label + short reasoning.
- **Keep the YOLO datasets as your fallback/v2 plan** — if the VLM approach is too slow or unreliable close to demo day, you already have clean labeled data ready to fine-tune YOLOv8n (n = nano, trains fast on limited compute, runs on-device if you ever want offline mode).
- This also sidesteps the biggest license/dataset risk: you're not training a production model on data with unclear provenance, you're using it as prompt examples and a validation/test set to *measure* how well the VLM approach performs before you commit to it.

**Validate the VLM approach quantitatively before demo day:** run your held-out test split (the 10% you set aside per class) through the prompted classifier and compute accuracy/confusion matrix per class. Don't trust it on vibes.

---

## 4. Backend (FastAPI — you already know this)

```
POST /analyze
  in:  multipart image
  out: { label: "burn_mild", confidence: 0.82, steps: [...], disclaimer: "..." }
```

Flow inside the endpoint:
1. Receive image, basic validation (size/type).
2. Call the vision-classification step (VLM call with few-shot prompt).
3. If confidence is low or label is "normal"/unclear → return a generic "can't confidently identify this, here's general first-aid guidance + call emergency services" response rather than guessing.
4. Feed the confirmed label into a second Claude call with a **fixed system prompt per emergency type** (burn/cut/choking) that constrains it to known, standard first-aid steps rather than freely generating medical advice — this matters both for safety and for demo reliability.
5. Append the disclaimer server-side (not model-generated) and return JSON.

Deploy target for a demo: a single small server (Render/Railway/Fly.io free tier, or even your own laptop on the venue Wi-Fi with ngrok as backup) — don't overbuild infra for a 2-3 scenario stage demo.

---

## 5. Database — for the demo, keep it minimal

You already used **Supabase** on WAVE, so it's the path of least friction if you need a DB at all. But be honest about whether you need one:

- **If the demo is just "point camera → get steps," you don't need a database at all.** No user accounts, no persistence — the whole flow can be stateless (image in, JSON out). This is the simplest and most demo-safe option; nothing to break on stage.
- **If you want to log demo runs / show a "history" screen** (nice-to-have, not required): use **Supabase (Postgres)** — one table (`scans`: id, timestamp, label, confidence, image_url) is enough. You already know the SDK from WAVE, so this is near-zero new learning cost.
- **Skip Firebase/Mongo/anything new** — not worth learning a new tool for a demo when Supabase already fits and you know it.

Recommendation: **build stateless first**, add the Supabase `scans` table only if you have time left over and want the "history/dashboard" nice-to-have for judges.

---

## 6. Mobile app (Flutter — you already know this)

Minimal screen set:
1. **Camera screen** — live camera preview (`camera` package), capture button.
2. **Loading state** — while waiting on the backend call (keep it under ~3-5s or the demo will drag; this is your key performance budget).
3. **Results screen** — label, confidence, numbered first-aid steps, and a persistent, high-contrast "Call Emergency Services" banner/button that's always visible, never scrollable-away.

Keep it to these three screens for the MVP. No auth, no onboarding, no settings — every extra screen is more surface area to break during a live demo.

---

## 7. Step-by-step build order

1. Clean and merge the three datasets into one label taxonomy + train/val/test split (§2).
2. Build and test the classification prompt against your test split; measure accuracy per class; iterate on the prompt (not the model) until it's reliable on your mock props.
3. Build the FastAPI `/analyze` endpoint end-to-end with the VLM call, tested via curl/Postman before touching the app.
4. Write and lock the first-aid step content per label with the LLM step — review this content carefully (or have it reviewed) since it's the actual advice being shown.
5. Build the Flutter camera → upload → display flow against the live backend.
6. Rehearse the exact 2–3 mock scenarios repeatedly, on the actual venue Wi-Fi if possible, with the actual props.
7. Only after the above works reliably: add the Supabase history table if time allows.

---

## 8. Likely issues — and how to avoid them ahead of time

- **Network flakiness on stage.** Live demos die from bad venue Wi-Fi more than bad models. Mitigation: have a **local fallback mode** — cache 2-3 known-good request/response pairs in the app and a manual "offline demo" toggle that replays them if the live call times out.
- **Latency.** A VLM round-trip can take a few seconds; if it's noticeably slow, the demo feels broken even if it's "working." Mitigation: set a hard timeout (e.g., 6s) with a friendly fallback message, and pre-warm the connection before you go on stage (send a dummy request a minute before).
- **Vision model misclassifying your specific mock props.** Fake theatrical burns/cuts often don't look like real ones (props read as "normal skin" or misclassify). Mitigation: test the *exact* props you'll use on stage during development, not just dataset images — this is your single highest-risk failure point.
- **Overconfident wrong answers.** A model that's very confident about the wrong label is worse than one that says "not sure." Mitigation: set a confidence threshold; below it, always show the "can't confidently identify — call emergency services" fallback response instead of guessing.
- **Choking dataset is small and unvetted** (from our earlier research — ~1,200 images, unpublished collection methodology). Don't fully trust its accuracy numbers; validate specifically on your own choking mock scenario photos before demo day.
- **Liability/framing.** This app gives medical guidance to laypeople — keep the "call emergency services" disclaimer non-dismissible and shown before the steps, not after, and avoid any wording that implies diagnosis ("you have a second-degree burn") vs. guidance ("this looks like it may be a burn — here's what to do while you call for help").
- **License attribution.** Both datasets you're using (MiliTech cuts, ChokeDetection) are CC BY 4.0 — credit the original authors (Mapua University Makati; Hezy) in your slides/README. Cheap to do, embarrassing to forget in front of judges.
- **Scope creep.** It's tempting to add classes, screens, or the Supabase history feature before the core flow is bulletproof. Lock the three-screen, one-endpoint MVP first; treat everything else as stretch goals only after a full rehearsal succeeds twice in a row.
