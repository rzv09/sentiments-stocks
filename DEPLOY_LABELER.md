# Deploying the labeler (v0.2, 3-class)

Needs Docker and AWS credentials with ECR push + `lambda:UpdateFunctionCode`.
The `finance-wiki-reader` IAM user does **not** have these, by design — use your
admin profile.

The `.joblib` is gitignored, so a fresh clone has no model. **Train first.**

## 1. Train

```bash
python scripts/train_baseline.py          # writes ml/models/baseline/
cat ml/models/baseline/MODEL_VERSION      # must read: baseline-0.2-3class
```

The dataset (`data/raw/Sentences_AllAgree.txt`) is also gitignored. It is the
Financial PhraseBank, CC BY-NC-SA 3.0 — non-commercial use only.

## 2. Build

```bash
docker buildx build --platform linux/amd64 \
  -t sentiments-labeler:v0.2 -f lambda_labeler/Dockerfile . --load
```

Build context is the repo root, not `lambda_labeler/`.

Cold start verifies the artifact's `MODEL_VERSION` against the string the code
stamps on rows, and refuses to start on a mismatch. Shipping a stale model under
a new version string mislabels data silently and downstream consumers trust the
stamp, so this is a hard failure rather than a warning.

## 3. Smoke-test locally before pushing

```bash
docker run --rm --platform linux/amd64 \
  -v ~/.aws:/root/.aws -e AWS_REGION=us-east-1 \
  -e TABLE_NAME=SentimentsStocksRawNews -e BATCH_LIMIT=5 \
  --entrypoint /var/lang/bin/python3.11 sentiments-labeler:v0.2 \
  -c "import app; print(app.handler({}, None))"
```

Expect `Model version verified: baseline-0.2-3class` in the logs. `BATCH_LIMIT=5`
keeps the blast radius small; `update_item` only writes rows where `sentiment`
is absent, so this cannot overwrite existing data.

## 4. Push and update

```bash
ACCOUNT=563171768646; REGION=us-east-1; REPO=sentiments-labeler
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker tag sentiments-labeler:v0.2 $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:v0.2
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:v0.2

aws lambda update-function-code --function-name <labeler-fn-name> \
  --image-uri $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:v0.2
```

Tag a real version rather than `:latest`, so a rollback is a one-line re-point
at the previous tag.

## 5. Environment variables

- `THRESHOLD` — now unused. A 3-class argmax has no single cutoff. Harmless to
  leave; delete it to avoid implying it still does something.
- `MODEL_VERSION` — defaults to `baseline-0.2-3class`. Only set it to override,
  and note the cold-start guard compares against it.

## 6. After deploying

New rows carry `sentiment` (which can now be `neutral`), `confidence`,
`sentiment_score` and `model_version`. Existing rows are untouched.

The two generations are on different scales and **must not be pooled** — the
wiki's `query_news_db.dominant_model()` enforces this, aggregating over one
generation and reporting what it excluded.

Then, in the wiki repo, after ~a week of v0.2 rows:

1. Recalibrate `SIGNAL_THRESHOLDS["baseline-0.2-3class"]` in
   `tools/_weekly_helper.py`. The current values are provisional, set from a
   ten-headline spot check before any v0.2 row existed.
2. Re-baseline drift. Every baseline computed before the rollout is v0.1 and
   does not carry over.

## Rollback

Re-point the function at the previous image tag. Rows already written keep their
`model_version` stamp, so mixed data stays readable — that is what the stamp is
for.

## Not done here

No backfill. Re-scoring the existing corpus under v0.2 would make history
self-consistent but is a large irreversible write, and `update_item`'s
`attribute_not_exists(sentiment)` condition deliberately prevents it happening
as a side effect of a routine run.
