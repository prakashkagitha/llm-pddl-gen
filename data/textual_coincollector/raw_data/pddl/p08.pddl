(define (problem coin_collector_numLocations5_numDistractorItems0_seed7)
  (:domain coin-collector)
  (:objects
    kitchen pantry backyard corridor bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen pantry south)
    (closed-door kitchen backyard west)
    (closed-door pantry kitchen north)
    (closed-door backyard kitchen east)
    (closed-door backyard corridor west)
    (closed-door corridor backyard east)
    (closed-door corridor bedroom west)
    (closed-door bedroom corridor east)
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