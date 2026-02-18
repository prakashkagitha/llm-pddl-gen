(define (problem coin_collector_numLocations3_numDistractorItems0_seed84)
  (:domain coin-collector)
  (:objects
    kitchen pantry corridor - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen pantry south)
    (connected kitchen corridor west)
    (closed-door pantry kitchen north)
    (connected corridor kitchen east)
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