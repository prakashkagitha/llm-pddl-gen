(define (problem coin_collector_numLocations5_numDistractorItems0_seed96)
  (:domain coin-collector)
  (:objects
    kitchen corridor backyard pantry bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (connected kitchen corridor north)
    (closed-door kitchen backyard south)
    (closed-door kitchen pantry east)
    (connected corridor kitchen south)
    (closed-door corridor bedroom east)
    (closed-door backyard kitchen north)
    (closed-door pantry kitchen west)
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