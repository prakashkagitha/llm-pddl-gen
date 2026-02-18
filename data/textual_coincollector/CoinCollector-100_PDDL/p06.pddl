(define (problem coin_collector_numLocations5_numDistractorItems0_seed83)
  (:domain coin-collector)
  (:objects
    kitchen backyard pantry corridor bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen backyard north)
    (closed-door kitchen pantry east)
    (connected kitchen corridor west)
    (closed-door backyard kitchen south)
    (closed-door pantry kitchen west)
    (connected corridor kitchen east)
    (closed-door corridor bedroom west)
    (closed-door bedroom corridor east)
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