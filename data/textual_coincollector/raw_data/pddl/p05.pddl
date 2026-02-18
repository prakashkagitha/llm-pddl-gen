(define (problem coin_collector_numLocations5_numDistractorItems0_seed62)
  (:domain coin-collector)
  (:objects
    kitchen pantry corridor bedroom backyard - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen pantry south)
    (connected kitchen corridor east)
    (closed-door pantry kitchen north)
    (connected corridor kitchen west)
    (closed-door corridor bedroom north)
    (closed-door corridor backyard east)
    (closed-door bedroom corridor south)
    (closed-door backyard corridor west)
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