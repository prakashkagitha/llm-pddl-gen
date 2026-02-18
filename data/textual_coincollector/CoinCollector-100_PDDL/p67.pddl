(define (problem coin_collector_numLocations5_numDistractorItems0_seed70)
  (:domain coin-collector)
  (:objects
    kitchen corridor pantry bedroom backyard - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (connected kitchen corridor north)
    (closed-door kitchen pantry east)
    (connected corridor kitchen south)
    (closed-door corridor bedroom north)
    (closed-door corridor backyard west)
    (closed-door pantry kitchen west)
    (closed-door bedroom corridor south)
    (closed-door backyard corridor east)
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