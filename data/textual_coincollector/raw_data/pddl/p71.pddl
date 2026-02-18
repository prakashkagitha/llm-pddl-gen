(define (problem coin_collector_numLocations7_numDistractorItems0_seed30)
  (:domain coin-collector)
  (:objects
    kitchen pantry backyard living_room bedroom bathroom corridor - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen pantry south)
    (closed-door kitchen backyard west)
    (closed-door pantry kitchen north)
    (closed-door backyard kitchen east)
    (closed-door backyard living_room north)
    (closed-door living_room backyard south)
    (closed-door living_room bedroom east)
    (closed-door living_room bathroom west)
    (closed-door bedroom living_room west)
    (closed-door bathroom living_room east)
    (closed-door bathroom corridor west)
    (closed-door corridor bathroom east)
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