(define (problem coin_collector_numLocations7_numDistractorItems0_seed64)
  (:domain coin-collector)
  (:objects
    kitchen pantry backyard living_room corridor bedroom bathroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen pantry north)
    (closed-door kitchen backyard south)
    (connected kitchen living_room west)
    (closed-door pantry kitchen south)
    (closed-door backyard kitchen north)
    (closed-door backyard corridor south)
    (connected living_room kitchen east)
    (closed-door living_room bedroom west)
    (closed-door corridor backyard north)
    (closed-door corridor bathroom east)
    (closed-door bedroom living_room east)
    (closed-door bathroom corridor west)
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