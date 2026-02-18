(define (problem coin_collector_numLocations5_numDistractorItems0_seed74)
  (:domain coin-collector)
  (:objects
    kitchen backyard corridor pantry bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen backyard north)
    (connected kitchen corridor east)
    (closed-door kitchen pantry west)
    (closed-door backyard kitchen south)
    (connected corridor kitchen west)
    (closed-door corridor bedroom east)
    (closed-door pantry kitchen east)
    (closed-door bedroom corridor west)
    (location coin corridor)
    (is-reverse north south)
    (is-reverse south north)
    (is-reverse east west)
    (is-reverse west east)
  )
  (:goal 
    (taken coin)
  )
)