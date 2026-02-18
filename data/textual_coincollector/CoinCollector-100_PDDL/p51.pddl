(define (problem coin_collector_numLocations7_numDistractorItems0_seed75)
  (:domain coin-collector)
  (:objects
    kitchen backyard pantry living_room corridor bathroom bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen backyard south)
    (closed-door kitchen pantry east)
    (closed-door backyard kitchen north)
    (closed-door backyard living_room south)
    (closed-door backyard corridor west)
    (closed-door pantry kitchen west)
    (closed-door living_room backyard north)
    (closed-door living_room bathroom south)
    (closed-door living_room bedroom west)
    (closed-door corridor backyard east)
    (closed-door corridor bedroom south)
    (closed-door bathroom living_room north)
    (closed-door bedroom living_room east)
    (closed-door bedroom corridor north)
    (location coin kitchen)
    (is-reverse north south)
    (is-reverse south north)
    (is-reverse east west)
    (is-reverse west east)
  )
  (:goal 
    (taken coin)
  )
)