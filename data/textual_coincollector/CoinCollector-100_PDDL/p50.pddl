(define (problem coin_collector_numLocations7_numDistractorItems0_seed8)
  (:domain coin-collector)
  (:objects
    kitchen pantry backyard corridor living_room bathroom bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen pantry north)
    (closed-door kitchen backyard south)
    (closed-door pantry kitchen south)
    (closed-door backyard kitchen north)
    (closed-door backyard corridor south)
    (closed-door backyard living_room west)
    (closed-door corridor backyard north)
    (closed-door corridor bathroom west)
    (closed-door living_room backyard east)
    (closed-door living_room bedroom north)
    (closed-door living_room bathroom south)
    (closed-door bathroom corridor east)
    (closed-door bathroom living_room north)
    (closed-door bedroom living_room south)
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