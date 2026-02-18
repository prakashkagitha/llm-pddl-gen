(define (problem coin_collector_numLocations7_numDistractorItems0_seed38)
  (:domain coin-collector)
  (:objects
    kitchen pantry backyard living_room corridor bathroom bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen pantry south)
    (closed-door kitchen backyard west)
    (closed-door pantry kitchen north)
    (closed-door backyard kitchen east)
    (closed-door backyard living_room west)
    (closed-door living_room backyard east)
    (connected living_room corridor west)
    (connected corridor living_room east)
    (closed-door corridor bathroom north)
    (closed-door corridor bedroom south)
    (closed-door bathroom corridor south)
    (closed-door bedroom corridor north)
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