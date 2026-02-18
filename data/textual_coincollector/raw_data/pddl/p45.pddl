(define (problem coin_collector_numLocations5_numDistractorItems0_seed52)
  (:domain coin-collector)
  (:objects
    kitchen pantry backyard corridor bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen pantry east)
    (closed-door kitchen backyard west)
    (closed-door pantry kitchen west)
    (closed-door backyard kitchen east)
    (closed-door backyard corridor south)
    (closed-door corridor backyard north)
    (closed-door corridor bedroom east)
    (closed-door bedroom corridor west)
    (location coin bedroom)
    (is-reverse north south)
    (is-reverse south north)
    (is-reverse east west)
    (is-reverse west east)
  )
  (:goal 
    (taken coin)
  )
)