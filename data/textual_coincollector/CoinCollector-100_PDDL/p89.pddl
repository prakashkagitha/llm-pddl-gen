(define (problem coin_collector_numLocations7_numDistractorItems0_seed63)
  (:domain coin-collector)
  (:objects
    kitchen living_room pantry backyard corridor bedroom bathroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (connected kitchen living_room south)
    (closed-door kitchen pantry east)
    (closed-door kitchen backyard west)
    (connected living_room kitchen north)
    (closed-door pantry kitchen west)
    (closed-door backyard kitchen east)
    (closed-door backyard corridor west)
    (closed-door corridor backyard east)
    (closed-door corridor bedroom north)
    (closed-door corridor bathroom south)
    (closed-door bedroom corridor south)
    (closed-door bathroom corridor north)
    (location coin corridor)
    (is-reverse north south)
    (is-reverse south north)
    (is-reverse east west)
    (is-reverse west east)
  )
  (:goal 
    (taken coin)
  )
)