{{ config(materialized='view') }}

/*
  Staging model: lê bronze.products_raw e aplica limpezas básicas.
  Não faz joins nem cálculos pesados — só tipagem e normalização.
*/

select
    product_id,

    -- limpeza de texto
    trim(title)                          as title,
    trim(description)                    as description,
    lower(trim(category))                as category,

    -- tipagem explícita
    price::numeric(10, 2)                as price,
    image,
    rating_rate::numeric(3, 2)           as rating_rate,
    rating_count::integer                as rating_count,

    -- metadado de carga
    current_timestamp                    as _loaded_at

from {{ source('bronze', 'products_raw') }}
where product_id is not null
