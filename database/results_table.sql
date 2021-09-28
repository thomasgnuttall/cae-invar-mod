DROP TABLE results;
CREATE TABLE results (
   patternNumber BIGINT PRIMARY KEY,
   recordingId CHAR(100) NOT NULL,
   elementNumber INT NOT NULL,
   durationElements FLOAT NOT NULL,
   startTimeSeconds FLOAT NOT NULL,
   durationSeconds FLOAT NOT NULL,
   patternGroup INT NOT NULL,
   rankInGroup INT NOT NULL
);


DROP TABLE IF EXISTS similarity;
CREATE TABLE similarity (
   patternNumberOne BIGINT,
   patternNumberTwo BIGINT,
   similarityName CHAR(100),
   similarity FLOAT
);
