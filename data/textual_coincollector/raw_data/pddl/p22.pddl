(define (problem coin_collector_numLocations3_numDistractorItems0_seed23)
  (:domain coin-collector)
  (:objects
    kitchen corridor pantry - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (connected kitchen corridor east)
    (closed-door kitchen pantry west)
    (connected corridor kitchen west)
    (closed-door pantry kitchen east)
    (location coin kitchen)
    (is-reverse north south)
    (is-reverse south north)
    (is-reverse east west)
    (is-reverse west east)
  )
  (:goal 
    (taken coin)
  )
)