(define (problem coin_collector_numLocations5_numDistractorItems0_seed93)
  (:domain coin-collector)
  (:objects
    kitchen corridor pantry backyard bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (connected kitchen corridor south)
    (closed-door kitchen pantry east)
    (closed-door kitchen backyard west)
    (connected corridor kitchen north)
    (closed-door corridor bedroom east)
    (closed-door pantry kitchen west)
    (closed-door backyard kitchen east)
    (closed-door bedroom corridor west)
    (location coin pantry)
    (is-reverse north south)
    (is-reverse south north)
    (is-reverse east west)
    (is-reverse west east)
  )
  (:goal 
    (taken coin)
  )
)