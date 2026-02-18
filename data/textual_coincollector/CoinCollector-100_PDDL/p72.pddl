(define (problem coin_collector_numLocations7_numDistractorItems0_seed11)
  (:domain coin-collector)
  (:objects
    kitchen corridor pantry living_room backyard bedroom bathroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (connected kitchen corridor north)
    (closed-door kitchen pantry west)
    (connected corridor kitchen south)
    (connected corridor living_room north)
    (closed-door corridor backyard east)
    (closed-door pantry kitchen east)
    (connected living_room corridor south)
    (closed-door living_room bedroom west)
    (closed-door backyard corridor west)
    (closed-door bedroom living_room east)
    (connected bedroom bathroom north)
    (connected bathroom bedroom south)
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