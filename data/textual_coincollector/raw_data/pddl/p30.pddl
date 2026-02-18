(define (problem coin_collector_numLocations7_numDistractorItems0_seed24)
  (:domain coin-collector)
  (:objects
    kitchen living_room corridor backyard pantry bedroom bathroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (connected kitchen living_room north)
    (connected kitchen corridor south)
    (closed-door kitchen backyard east)
    (closed-door kitchen pantry west)
    (connected living_room kitchen south)
    (closed-door living_room bedroom north)
    (connected corridor kitchen north)
    (closed-door corridor bathroom west)
    (closed-door backyard kitchen west)
    (closed-door pantry kitchen east)
    (closed-door bedroom living_room south)
    (closed-door bathroom corridor east)
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