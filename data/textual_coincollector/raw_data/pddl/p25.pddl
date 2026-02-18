(define (problem coin_collector_numLocations5_numDistractorItems0_seed73)
  (:domain coin-collector)
  (:objects
    kitchen backyard pantry corridor bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen backyard south)
    (closed-door kitchen pantry east)
    (closed-door backyard kitchen north)
    (closed-door backyard corridor south)
    (closed-door pantry kitchen west)
    (closed-door corridor backyard north)
    (closed-door corridor bedroom west)
    (closed-door bedroom corridor east)
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