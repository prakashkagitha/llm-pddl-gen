(define (problem coin_collector_numLocations7_numDistractorItems0_seed35)
  (:domain coin-collector)
  (:objects
    kitchen backyard corridor pantry living_room bedroom bathroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen backyard north)
    (connected kitchen corridor south)
    (closed-door kitchen pantry west)
    (closed-door backyard kitchen south)
    (closed-door backyard living_room north)
    (connected corridor kitchen north)
    (closed-door corridor bedroom east)
    (closed-door corridor bathroom west)
    (closed-door pantry kitchen east)
    (closed-door living_room backyard south)
    (closed-door bedroom corridor west)
    (closed-door bathroom corridor east)
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