(define (problem coin_collector_numLocations5_numDistractorItems0_seed2)
  (:domain coin-collector)
  (:objects
    kitchen pantry backyard corridor bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen pantry north)
    (closed-door kitchen backyard east)
    (closed-door pantry kitchen south)
    (closed-door backyard kitchen west)
    (closed-door backyard corridor south)
    (closed-door corridor backyard north)
    (closed-door corridor bedroom west)
    (closed-door bedroom corridor east)
    (location coin kitchen)
    (is-reverse north south)
    (is-reverse south north)
    (is-reverse east west)
    (is-reverse west east)
  )
  (:goal 
    (taken coin)
  )
)