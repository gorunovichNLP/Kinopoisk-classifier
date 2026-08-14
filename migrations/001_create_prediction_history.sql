BEGIN;

CREATE SCHEMA IF NOT EXISTS sentiment;

CREATE TABLE sentiment.prediction_history (
    prediction_id      text                     PRIMARY KEY,
    schema_version     smallint                 NOT NULL,
    source_event_id    text                     NOT NULL,
    review_id          text                     NOT NULL,
    movie_id           text,
    source_created_at  timestamp with time zone,
    sentiment          text                     NOT NULL,
    label_id           smallint                 NOT NULL,
    confidence         double precision         NOT NULL,
    probability_neg    double precision         NOT NULL,
    probability_neu    double precision         NOT NULL,
    probability_pos    double precision         NOT NULL,
    model_name         text                     NOT NULL,
    model_version      text                     NOT NULL,
    model_run_id       text,
    predicted_at       timestamp with time zone NOT NULL,
    stored_at          timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT prediction_history_schema_version_v1
        CHECK (schema_version = 1),
    CONSTRAINT prediction_history_prediction_id_sha256
        CHECK (prediction_id ~ '^[0-9a-f]{64}$'),
    CONSTRAINT prediction_history_review_id_object_id
        CHECK (review_id ~ '^[0-9a-f]{24}$'),
    CONSTRAINT prediction_history_source_event_id
        CHECK (source_event_id = 'review:' || review_id),
    CONSTRAINT prediction_history_movie_id_non_blank
        CHECK (movie_id IS NULL OR btrim(movie_id) <> ''),
    CONSTRAINT prediction_history_sentiment
        CHECK (sentiment IN ('neg', 'neu', 'pos')),
    CONSTRAINT prediction_history_label
        CHECK (label_id BETWEEN 0 AND 2),
    CONSTRAINT prediction_history_label_mapping
        CHECK (
            (label_id = 0 AND sentiment = 'neg') OR
            (label_id = 1 AND sentiment = 'neu') OR
            (label_id = 2 AND sentiment = 'pos')
        ),
    CONSTRAINT prediction_history_confidence_range
        CHECK (confidence BETWEEN 0.0 AND 1.0),
    CONSTRAINT prediction_history_probability_neg_range
        CHECK (probability_neg BETWEEN 0.0 AND 1.0),
    CONSTRAINT prediction_history_probability_neu_range
        CHECK (probability_neu BETWEEN 0.0 AND 1.0),
    CONSTRAINT prediction_history_probability_pos_range
        CHECK (probability_pos BETWEEN 0.0 AND 1.0),
    CONSTRAINT prediction_history_probability_sum
        CHECK (abs(probability_neg + probability_neu + probability_pos - 1.0) <= 0.000001),
    CONSTRAINT prediction_history_confidence_matches_label
        CHECK (
            abs(
                confidence - CASE label_id
                    WHEN 0 THEN probability_neg
                    WHEN 1 THEN probability_neu
                    WHEN 2 THEN probability_pos
                END
            ) <= 0.000001
        ),
    CONSTRAINT prediction_history_model_name_non_blank
        CHECK (btrim(model_name) <> ''),
    CONSTRAINT prediction_history_model_version_non_blank
        CHECK (btrim(model_version) <> ''),
    CONSTRAINT prediction_history_model_run_id_non_blank
        CHECK (model_run_id IS NULL OR btrim(model_run_id) <> ''),
    CONSTRAINT prediction_history_model_identity
        UNIQUE (source_event_id, model_name, model_version)
);

CREATE INDEX prediction_history_review_predicted_at_idx
    ON sentiment.prediction_history (review_id, predicted_at DESC);

CREATE INDEX prediction_history_model_idx
    ON sentiment.prediction_history (model_name, model_version);

CREATE INDEX prediction_history_sentiment_idx
    ON sentiment.prediction_history (sentiment);

CREATE VIEW sentiment.latest_review_prediction AS
SELECT DISTINCT ON (review_id)
    prediction_id,
    schema_version,
    source_event_id,
    review_id,
    movie_id,
    source_created_at,
    sentiment,
    label_id,
    confidence,
    probability_neg,
    probability_neu,
    probability_pos,
    model_name,
    model_version,
    model_run_id,
    predicted_at,
    stored_at
FROM sentiment.prediction_history
ORDER BY review_id, predicted_at DESC, stored_at DESC, prediction_id DESC;

COMMIT;
