

## Added uuid col in my upstream table
create EXTENSION if not EXISTS "uuid-ossp";
-- Alter table apparels add column uuid UUID;
UPDATE apparels SET uuid = uuid_generate_v4();



## Created my own table
DROP TABLE documents;
CREATE TABLE documents AS (SELECT * FROM apparels);
SELECT * FROM documents LIMIT 1000;


## Table is unstable while generating embeddings
SELECT count(*) FROM documents
  WHERE vector_dims(embedding) <= 0 or embedding is NULL;
