{{ config(materialized='table') }}

select
  product_id,
  title,
  category_clean as category,
  price,
  image,
  trim(concat_ws(' ', title, category_clean, description_clean)) as full_text
from {{ ref('products_clean') }}
