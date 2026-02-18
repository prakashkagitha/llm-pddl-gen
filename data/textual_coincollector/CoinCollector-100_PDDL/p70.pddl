(define (problem coin_collector_numLocations7_numDistractorItems0_seed92)
  (:domain coin-collector)
  (:objects
    kitchen living_room corridor pantry backyard bathroom bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (connected kitchen living_room north)
    (connected kitchen corridor east)
    (closed-door kitchen pantry west)
    (connected living_room kitchen south)
    (closed-door living_room backyard north)
    (closed-door living_room bathroom east)
    (closed-door living_room bedroom west)
    (connected corridor kitchen west)
    (closed-door corridor bathroom north)
    (closed-door pantry kitchen east)
    (closed-door backyard living_room south)
    (closed-door bathroom living_room west)
    (closed-door bathroom corridor south)
    (closed-door bedroom living_room east)
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