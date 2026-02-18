(define (problem coin_collector_numLocations7_numDistractorItems0_seed50)
  (:domain coin-collector)
  (:objects
    kitchen pantry backyard corridor living_room bedroom bathroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen pantry north)
    (closed-door kitchen backyard south)
    (connected kitchen corridor west)
    (closed-door pantry kitchen south)
    (closed-door backyard kitchen north)
    (closed-door backyard living_room east)
    (connected corridor kitchen east)
    (closed-door living_room backyard west)
    (closed-door living_room bedroom north)
    (closed-door living_room bathroom south)
    (closed-door bedroom living_room south)
    (closed-door bathroom living_room north)
    (location coin bedroom)
    (is-reverse north south)
    (is-reverse south north)
    (is-reverse east west)
    (is-reverse west east)
  )
  (:goal 
    (taken coin)
  )
)