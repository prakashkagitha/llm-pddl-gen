(define (problem coin_collector_numLocations7_numDistractorItems0_seed1)
  (:domain coin-collector)
  (:objects
    kitchen bathroom backyard pantry corridor living_room bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen bathroom south)
    (closed-door kitchen backyard east)
    (closed-door kitchen pantry west)
    (closed-door bathroom kitchen north)
    (closed-door bathroom corridor east)
    (closed-door backyard kitchen west)
    (closed-door backyard corridor south)
    (closed-door backyard living_room east)
    (closed-door pantry kitchen east)
    (closed-door corridor bathroom west)
    (closed-door corridor backyard north)
    (closed-door living_room backyard west)
    (closed-door living_room bedroom east)
    (closed-door bedroom living_room west)
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