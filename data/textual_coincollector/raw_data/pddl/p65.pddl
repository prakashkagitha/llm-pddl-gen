(define (problem coin_collector_numLocations5_numDistractorItems0_seed72)
  (:domain coin-collector)
  (:objects
    kitchen backyard pantry corridor bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen backyard south)
    (closed-door kitchen pantry west)
    (closed-door backyard kitchen north)
    (closed-door backyard corridor east)
    (closed-door pantry kitchen east)
    (closed-door corridor backyard west)
    (closed-door corridor bedroom east)
    (closed-door bedroom corridor west)
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