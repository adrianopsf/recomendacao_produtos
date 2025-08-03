{{ config(materialized='view') }}

with src as (
  select
    product_id,
    title,
    description,
    category,
    price,
    image
  from {{ source('bronze','products_raw') }}
)

select
  product_id,
  title,
  trim(lower(coalesce(description,''))) as description_clean,
  trim(lower(coalesce(category,'')))    as category_clean,
  price,
  image
from src
