(define (problem coin_collector_numLocations3_numDistractorItems0_seed71)
  (:domain coin-collector)
  (:objects
    kitchen corridor pantry - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (connected kitchen corridor south)
    (closed-door kitchen pantry east)
    (connected corridor kitchen north)
    (closed-door pantry kitchen west)
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