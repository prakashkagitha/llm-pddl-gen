(define (problem coin_collector_numLocations5_numDistractorItems0_seed25)
  (:domain coin-collector)
  (:objects
    kitchen pantry corridor backyard bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen pantry north)
    (connected kitchen corridor east)
    (closed-door kitchen backyard west)
    (closed-door pantry kitchen south)
    (connected corridor kitchen west)
    (closed-door corridor bedroom east)
    (closed-door backyard kitchen east)
    (closed-door bedroom corridor west)
    (location coin backyard)
    (is-reverse north south)
    (is-reverse south north)
    (is-reverse east west)
    (is-reverse west east)
  )
  (:goal 
    (taken coin)
  )
)