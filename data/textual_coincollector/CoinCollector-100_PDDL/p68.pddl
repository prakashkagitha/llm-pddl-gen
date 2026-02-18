(define (problem coin_collector_numLocations5_numDistractorItems0_seed5)
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
    (closed-door backyard corridor west)
    (closed-door pantry kitchen west)
    (closed-door corridor backyard east)
    (closed-door corridor bedroom north)
    (closed-door bedroom corridor south)
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