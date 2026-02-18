(define (problem coin_collector_numLocations5_numDistractorItems0_seed66)
  (:domain coin-collector)
  (:objects
    kitchen backyard corridor pantry bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen backyard north)
    (connected kitchen corridor south)
    (closed-door kitchen pantry east)
    (closed-door backyard kitchen south)
    (connected corridor kitchen north)
    (closed-door corridor bedroom west)
    (closed-door pantry kitchen west)
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