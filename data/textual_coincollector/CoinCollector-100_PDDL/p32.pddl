(define (problem coin_collector_numLocations7_numDistractorItems0_seed3)
  (:domain coin-collector)
  (:objects
    kitchen backyard pantry corridor living_room bathroom bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen backyard east)
    (closed-door kitchen pantry west)
    (closed-door backyard kitchen west)
    (closed-door backyard corridor north)
    (closed-door backyard living_room south)
    (closed-door pantry kitchen east)
    (closed-door corridor backyard south)
    (closed-door corridor bathroom north)
    (closed-door corridor bedroom west)
    (closed-door living_room backyard north)
    (closed-door bathroom corridor south)
    (closed-door bedroom corridor east)
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